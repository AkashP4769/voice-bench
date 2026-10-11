"""Benchmark orchestration.

One audio file is normalized once, then every selected provider transcribes the
identical bytes. Results are scored against a reference transcript when one is
supplied, and the whole run is persisted with the environment that produced it.

Providers run one after another, never in parallel: both engines are
single-model-per-process and Phonon serialises decodes behind one lock, so
concurrent runs would measure contention rather than the models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.audio.prepare import prepare_audio
from app.config import Settings
from app.domain.models import TranscriptionOptions, TranscriptionResult
from app.domain.provider import ProviderCapabilities
from app.environment import collect as collect_environment
from app.errors import ProviderError, ProviderUnavailable, ProviderTimeout
from app.providers.registry import capabilities_for
from app.scoring import metrics
from app.scoring.normalize import DEFAULT_POLICY
from app.storage.repository import Repository
from app.workers.pool import WorkerPool

logger = logging.getLogger(__name__)


@dataclass
class RunRequest:
    audio_path: Path
    providers: list[str]
    options: TranscriptionOptions = field(default_factory=TranscriptionOptions)
    reference_text: str | None = None
    audio_name: str | None = None
    notes: str | None = None
    score_policy: NormalizationPolicy = DEFAULT_POLICY


def _error_result(provider: str, model: str, message: str) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "transport": None,
        "text": "",
        "language": None,
        "words": [],
        "model_version": None,
        "audio_duration_ms": None,
        "latency_ms": None,
        "time_to_first_token_ms": None,
        "real_time_factor": None,
        "windowing": None,
        "truncated": None,
        "warnings": [],
        "error": message,
        "raw_response": {},
        "score": None,
    }


def _check_capabilities(
    capabilities: ProviderCapabilities, options: TranscriptionOptions
) -> str | None:
    problems = capabilities.accepts(options)
    if problems:
        return "; ".join(problems)
    return None


def execute_run(
    settings: Settings,
    pool: WorkerPool,
    repository: Repository,
    request: RunRequest,
) -> dict[str, Any]:
    """Run every requested provider over one file and persist the outcome."""
    prepared = prepare_audio(
        request.audio_path,
        settings.audio_dir,
        settings.defaults.target_sample_rate,
    )

    results: list[dict[str, Any]] = []
    for name in request.providers:
        config = settings.providers.get(name)
        if config is None:
            known = ", ".join(sorted(settings.providers)) or "none"
            results.append(
                _error_result(name, name, f"unknown provider {name!r} (known: {known})")
            )
            continue

        problem = _check_capabilities(capabilities_for(config), request.options)
        if problem is not None:
            # Fail before starting a worker: the options cannot be honoured, so
            # spending a model load on them would be waste.
            results.append(_error_result(name, config.model, problem))
            continue

        try:
            result = pool.run(name, prepared, request.options)
        except (ProviderError, ProviderUnavailable, ProviderTimeout) as exc:
            logger.warning("provider %s failed: %s", name, exc)
            results.append(_error_result(name, config.model, str(exc)))
            continue
        except KeyError as exc:
            results.append(_error_result(name, config.model, str(exc)))
            continue

        payload = result.as_dict()
        payload["score"] = None
        if request.reference_text and result.ok and result.text:
            score = metrics.score(
                request.reference_text, result.text, request.score_policy
            )
            payload["score"] = score.as_dict()
        results.append(payload)

    caveats = _run_caveats(settings, request, results)
    run = {
        "audio": {
            "path": str(prepared.path),
            "name": request.audio_name or Path(request.audio_path).name,
            "sha256": prepared.sha256,
            "duration_ms": prepared.duration_ms,
            "sample_rate": prepared.sample_rate,
            "channels": prepared.channels,
        },
        "providers": list(request.providers),
        "options": request.options.fingerprint(),
        "reference_text": request.reference_text,
        "environment": collect_environment(),
        "notes": request.notes,
        "caveats": caveats,
        "results": results,
    }
    run["id"] = repository.save_run(run)
    return run


def _run_caveats(
    settings: Settings,
    request: RunRequest,
    results: list[dict[str, Any]],
) -> list[str]:
    """Things a reader must know before comparing the numbers."""
    caveats: list[str] = []

    transports = {
        result["transport"] for result in results if result.get("transport")
    }
    if len(transports) > 1:
        caveats.append(
            "providers were reached through different transports "
            f"({', '.join(sorted(transports))}); latency is not directly comparable"
        )

    for result in results:
        if result.get("error"):
            continue
        windowing = result.get("windowing") or {}
        if windowing.get("window_count", 1) > 1:
            caveats.append(
                f"{result['provider']}: audio was split by the harness, so the "
                "transcript reflects our chunking as well as the model"
            )
        if result.get("truncated"):
            caveats.append(
                f"{result['provider']}: reported truncated output, so part of the "
                "audio may be missing"
            )

    if request.reference_text:
        caveats.append(
            f"word error rate uses the harness normalizer "
            f"({request.score_policy.value}), not Whisper's EnglishTextNormalizer"
        )
    else:
        caveats.append(
            "no reference transcript was supplied, so word error rate is not reported"
        )
    return caveats


def summarize_score(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the best provider by word error rate, when scores exist."""
    scored = [
        (result["provider"], result["score"])
        for result in results
        if result.get("score") and result["score"].get("wer") is not None
    ]
    if not scored:
        return None
    best_provider, best_score = min(scored, key=lambda item: item[1]["wer"])
    return {"best_wer_provider": best_provider, "best_wer": best_score["wer"]}
