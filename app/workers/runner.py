"""Provider worker process.

Owns exactly one model for its whole lifetime, because both engines allow only
one model per process:

* cactus-needle: "One model is loaded per process; it is not thread-safe."
* Fermion: one model is held by a single worker thread and every decode is
  serialised behind one lock.

Requests are therefore handled one at a time, in arrival order. Concurrency is
the pool's job, not this process's.

The protocol runs over a loopback socket rather than stdin/stdout. Both engines
print progress to stdout ("[fermion] fetching ...", tqdm bars), and redirecting
file descriptors around a native library turned out to be unreliable, so the
worker simply does not share a channel with engine noise: stdout and stderr are
pointed at a log file by the pool, and control traffic goes over the socket.
"""

from __future__ import annotations

import json
import logging
import socket
import sys
import traceback
from pathlib import Path
from typing import Any

from app.config import get_settings, load_settings
from app.errors import ProviderError, ProviderUnavailable, ProviderTimeout
from app.providers.registry import build_provider
from app.streaming.session import StreamOptions
from app.workers import protocol

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_S = 60.0


def _connect(port: int) -> socket.socket:
    return socket.create_connection(("127.0.0.1", port), timeout=CONNECT_TIMEOUT_S)


def _error_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "error": str(exc),
        "error_type": type(exc).__name__,
        "detail": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
    }


class _LiveSessionState:
    """Holds the single live session this worker may have open.

    Kept deliberately small: the worker loop is single-threaded, so a session is
    advanced one chunk at a time inline, and a slow decode naturally applies
    backpressure by leaving later chunks unread in the pipe.
    """

    def __init__(self) -> None:
        self.session = None
        self.session_id: str | None = None
        self.implementation: str | None = None

    @property
    def active(self) -> bool:
        return self.session is not None

    def start(self, provider, payload: dict[str, Any], send) -> None:
        session_id = payload.get("id")
        if self.active:
            self._fail(
                send,
                session_id,
                "a live session is already active for this provider",
            )
            return
        try:
            options = StreamOptions.from_wire(payload.get("options") or {})
            session = provider.open_stream(options)
        except Exception as exc:
            send(
                {"type": protocol.STREAM_ERROR, "id": session_id, **_error_payload(exc)}
            )
            return

        self.session = session
        self.session_id = session_id
        self.implementation = getattr(session, "implementation", None)
        send(
            {
                "type": protocol.STREAM_STARTED,
                "id": session_id,
                "implementation": self.implementation,
            }
        )

    def chunk(self, payload: dict[str, Any], send) -> None:
        session_id = payload.get("id")
        if self.session is None or session_id != self.session_id:
            self._fail(send, session_id, "no live session is open with that id")
            return
        try:
            samples = protocol.decode_audio_block(payload["audio"])
            events = self.session.feed(samples)
        except Exception as exc:
            # The session is unrecoverable after a feed failure: abandon it so a
            # later stop does not try to finalize a broken engine.
            self._abandon()
            send(
                {"type": protocol.STREAM_ERROR, "id": session_id, **_error_payload(exc)}
            )
            return
        send(
            {
                "type": protocol.STREAM_EVENTS,
                "id": session_id,
                "events": [event.to_wire() for event in events],
            }
        )

    def stop(self, payload: dict[str, Any], send) -> None:
        session_id = payload.get("id")
        session = self.session
        if session is None or session_id != self.session_id:
            self._fail(send, session_id, "no live session is open with that id")
            return

        self.session = None
        self.session_id = None
        try:
            result = session.stop()
        except Exception as exc:
            send(
                {"type": protocol.STREAM_ERROR, "id": session_id, **_error_payload(exc)}
            )
            return
        send(
            {"type": protocol.STREAM_DONE, "id": session_id, "result": result.as_dict()}
        )

    def close_quietly(self) -> None:
        """On shutdown, let a session finalize rather than leaving it dangling."""
        session, self.session = self.session, None
        self.session_id = None
        if session is not None:
            try:
                session.stop()
            except Exception:
                logger.debug("session did not stop cleanly on shutdown")

    def _abandon(self) -> None:
        self.session = None
        self.session_id = None

    @staticmethod
    def _fail(send, session_id: str | None, message: str) -> None:
        send(
            {
                "type": protocol.STREAM_ERROR,
                "id": session_id,
                "error": message,
                "error_type": ProviderError.__name__,
                "detail": "",
            }
        )


def _run_worker(provider_name: str, root: Path | None, port: int) -> int:
    settings = load_settings(root) if root else get_settings()
    conn = _connect(port)
    conn.settimeout(None)
    reader = conn.makefile("r", encoding="utf-8", newline="\n")
    writer = conn.makefile("w", encoding="utf-8", newline="\n")

    def send(payload: dict[str, Any]) -> None:
        writer.write(json.dumps(payload) + "\n")
        writer.flush()

    try:
        config = settings.provider(provider_name)
        provider = build_provider(config, settings)
    except Exception as exc:
        send({"type": protocol.FATAL, **_error_payload(exc)})
        return 2

    # Warm up before announcing readiness so the caller's first timed run
    # measures inference rather than a download, an unpack or a JIT compile.
    try:
        load_seconds = provider.warmup()
    except Exception as exc:
        send({"type": protocol.FATAL, **_error_payload(exc)})
        return 3

    send(
        {
            "type": protocol.READY,
            "provider": provider_name,
            "load_seconds": load_seconds,
            "describe": provider.describe(),
        }
    )

    # One live session at a time. Both engines keep stream state per process, so
    # a second session, or a batch job during a session, would corrupt it. The
    # API refuses these earlier for a better message; this is the enforcement
    # point that must hold regardless of what the API does.
    live = _LiveSessionState()

    for line in reader:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("worker %s received malformed input", provider_name)
            continue

        kind = payload.get("type")
        if kind == protocol.SHUTDOWN:
            live.close_quietly()
            return 0

        if kind == protocol.STREAM_START:
            live.start(provider, payload, send)
            continue
        if kind == protocol.STREAM_CHUNK:
            live.chunk(payload, send)
            continue
        if kind == protocol.STREAM_STOP:
            live.stop(payload, send)
            continue

        if kind != protocol.JOB:
            continue

        if live.active:
            send(
                {
                    "type": protocol.RESULT,
                    "id": payload.get("id"),
                    "ok": False,
                    "error": (
                        "a live session is active for this provider, so batch "
                        "transcription is unavailable until it ends. This is a "
                        "per-process engine limitation, not a queue."
                    ),
                    "error_type": ProviderError.__name__,
                    "detail": "",
                }
            )
            continue

        job = protocol.JobRequest.from_wire(payload)
        try:
            result = provider.transcribe(job.audio, job.options)
        except (ProviderError, ProviderUnavailable, ProviderTimeout) as exc:
            send(
                {
                    "type": protocol.RESULT,
                    "id": job.job_id,
                    "ok": False,
                    **_error_payload(exc),
                }
            )
            continue
        except Exception as exc:  # unexpected engine failure
            send(
                {
                    "type": protocol.RESULT,
                    "id": job.job_id,
                    "ok": False,
                    **_error_payload(exc),
                }
            )
            continue

        send(
            {
                "type": protocol.RESULT,
                "id": job.job_id,
                "ok": True,
                "result": result.as_dict(),
            }
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # argv: <provider> <port> [project-root]
    if len(argv) < 2:
        print(
            "usage: python -m app.workers.runner <provider> <port> [project-root]",
            file=sys.stderr,
        )
        return 2
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    provider_name = argv[0]
    port = int(argv[1])
    root = Path(argv[2]).resolve() if len(argv) > 2 else None
    try:
        return _run_worker(provider_name, root, port)
    except OSError as exc:
        # The pool closed or never accepted; nothing useful left to report.
        print(f"worker {provider_name}: socket error: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
