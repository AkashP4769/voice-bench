"""Realtime transcription over a WebSocket.

One socket carries everything: the browser sends a start frame, then raw PCM
frames, then a stop frame. The server fans the audio out to a thread per
provider and relays their events back, tagged with the provider they came from,
so several models can be watched side by side.
"""

from __future__ import annotations

import asyncio
import json
import logging

import numpy as np
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from app.api.deps import get_config, get_repository, get_worker_pool
from app.config import Settings
from app.providers.registry import capabilities_for
from app.streaming.events import StreamResult
from app.streaming.manager import StreamManager
from app.streaming.session import StreamOptions
from app.storage.repository import Repository

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stream"])

SAMPLE_RATE = 16000
MAX_PROVIDERS = 4


def _split(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


async def _reject(websocket: WebSocket, message: str) -> None:
    await websocket.send_json({"type": "error", "message": message})


async def _pump(websocket: WebSocket, manager: StreamManager) -> None:
    """Relay provider events to the browser as they arrive."""
    while True:
        kind, item = await manager.out.get()
        if kind != "event":
            # Results are delivered once, in the final payload, so a provider
            # finishing early does not produce a stray duplicate.
            continue
        await websocket.send_json(item.to_wire())


def _ended_payload(results: list[StreamResult]) -> dict:
    transcripts = {result.provider: result.text for result in results}
    metrics = {result.provider: result.metrics.as_dict() for result in results}
    errors = {
        result.provider: result.error for result in results if result.error
    }
    implementations = {
        result.provider: result.implementation for result in results
    }
    words = {result.provider: result.words for result in results if result.words}
    return {
        "type": "ended",
        "transcripts": transcripts,
        "metrics": metrics,
        "errors": errors,
        "implementations": implementations,
        "words": words,
    }


@router.websocket("/ws/realtime")
async def realtime_socket(websocket: WebSocket) -> None:
    await websocket.accept()

    settings = get_settings_from_socket()
    pool = get_worker_pool_direct(settings)
    loop = asyncio.get_running_loop()
    manager: StreamManager | None = None

    try:
        first = await websocket.receive()
        if first.get("type") == "websocket.disconnect":
            return

        raw = first.get("text")
        if not raw:
            await _reject(websocket, "the first message must be a start frame")
            return
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            await _reject(websocket, "the start frame was not valid JSON")
            return
        if payload.get("type") != "start":
            await _reject(websocket, "the first message must be a start frame")
            return

        providers = _split(payload.get("providers"))
        if not providers:
            await _reject(websocket, "select at least one provider")
            return
        if len(providers) > MAX_PROVIDERS:
            await _reject(
                websocket, f"at most {MAX_PROVIDERS} providers can run at once"
            )
            return
        unknown = [name for name in providers if name not in settings.providers]
        if unknown:
            known = ", ".join(sorted(settings.providers)) or "none"
            await _reject(
                websocket,
                f"unknown provider(s): {', '.join(unknown)} (known: {known})",
            )
            return

        unsupported = [
            name
            for name in providers
            if not capabilities_for(settings.provider(name)).streaming
        ]
        if len(unsupported) == len(providers):
            await _reject(
                websocket,
                "none of the selected providers can transcribe in realtime",
            )
            return

        declared_rate = payload.get("sampleRate")
        if declared_rate and int(declared_rate) != SAMPLE_RATE:
            # The browser resamples before sending, so a mismatch means a bug
            # rather than something to correct silently with a bad guess.
            await _reject(
                websocket,
                f"audio must be {SAMPLE_RATE} Hz mono float32; got {declared_rate}",
            )
            return

        options = StreamOptions(
            language=payload.get("language") or None,
            keywords=tuple(_split(payload.get("keywords"))),
            word_timestamps=bool(payload.get("wordTimestamps")),
            chunk_seconds=(
                float(payload["cadenceSeconds"])
                if isinstance(payload.get("cadenceSeconds"), (int, float))
                else None
            ),
        )

        manager = StreamManager(pool, providers, options, loop)
        # start() loads models, so it must not run on the event loop.
        await asyncio.to_thread(manager.start)

        await websocket.send_json(
            {
                "type": "ready",
                "providers": list(manager.implementations),
                "failed": manager.failed,
                "implementations": manager.implementations,
                "sampleRate": SAMPLE_RATE,
                "overrunBufferSeconds": 30,
            }
        )
        if not manager.implementations:
            await websocket.send_json(_ended_payload(manager.close()))
            return

        sender = asyncio.create_task(_pump(websocket, manager))
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                chunk = message.get("bytes")
                if chunk:
                    samples = np.frombuffer(chunk, dtype="<f4").astype(
                        np.float32, copy=False
                    )
                    manager.push(samples)
                    continue

                text = message.get("text")
                if not text:
                    continue
                try:
                    control = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if control.get("type") == "stop":
                    break
        finally:
            sender.cancel()

        results = await asyncio.to_thread(manager.close)
        manager = None
        await websocket.send_json(_ended_payload(results))
    except WebSocketDisconnect:
        logger.debug("realtime socket disconnected")
    except Exception as exc:  # never leave the browser without a reason
        logger.exception("realtime session failed")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        if manager is not None:
            try:
                await asyncio.to_thread(manager.close)
            except Exception:
                logger.debug("cleanup of a failed session did not complete")


@router.post("/api/realtime/save")
def save_realtime_session(
    payload: dict,
    settings: Settings = Depends(get_config),
    repository: Repository = Depends(get_repository),
) -> dict:
    """Persist a finished session as a run.

    Transcript text only: the session audio is not retained, so the stored run
    says so and cannot be re-scored. That is a deliberate choice, not an
    oversight, and it is recorded on the run.
    """
    providers = _split(payload.get("providers"))
    transcripts = payload.get("transcripts") or {}
    if not providers:
        return {"saved": False, "message": "no providers to save"}

    run = {
        "source": "realtime",
        "audio": None,
        "duration_ms": payload.get("durationMs"),
        "providers": providers,
        "options": {
            "language": payload.get("language"),
            "keywords": _split(payload.get("keywords")),
            "word_timestamps": bool(payload.get("wordTimestamps")),
            "cadence_seconds": payload.get("cadenceSeconds"),
        },
        "reference_text": payload.get("referenceText") or None,
        "notes": payload.get("notes") or None,
        "caveats": [
            "live session: the audio was not retained, so this run cannot be "
            "re-scored against a different reference or re-run",
        ],
        "results": [
            {
                "provider": name,
                "model": settings.providers[name].model
                if name in settings.providers
                else name,
                "transport": "stream",
                "text": transcripts.get(name) or "",
                "words": (payload.get("words") or {}).get(name) or [],
                "metrics": (payload.get("metrics") or {}).get(name) or {},
                "error": (payload.get("errors") or {}).get(name),
            }
            for name in providers
        ],
    }
    run_id = repository.save_realtime_run(run)
    return {"saved": True, "runId": run_id}


# ---------------------------------------------------------------------------
# Small accessors so the socket can reach configuration without FastAPI's
# dependency injection, which is not available on a WebSocket route the same way.
# ---------------------------------------------------------------------------


def get_settings_from_socket() -> Settings:
    from app.config import get_settings

    return get_settings()


def get_worker_pool_direct(settings: Settings):
    from app.workers.pool import get_pool

    return get_pool(settings)
