"""Running the evaluation set through the providers.

This reuses the same worker pool, audio normalization and scoring the single-file
benchmark uses, so an evaluation number and a one-off number are produced the same
way. What it adds is breadth: many clips, several languages, and an honest account
of the clip/provider pairs that were never attempted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from app.audio.prepare import prepare_audio
from app.config import Settings
from app.domain.models import TranscriptionOptions
from app.errors import ProviderError, ProviderUnavailable, ProviderTimeout
from app.evaluation.report import ERROR, OK, UNSUPPORTED, ClipResult
from app.evaluation.sources import Clip
from app.providers.registry import capabilities_for
from app.scoring import metrics
from app.scoring.normalize import DEFAULT_POLICY, NormalizationPolicy
from app.workers.pool import WorkerPool

logger = logging.getLogger(__name__)


@dataclass
class EvaluationOptions:
    providers: list[str]
    policy: NormalizationPolicy = DEFAULT_POLICY
    #: Run a provider on a language it does not declare, to observe the failure.
    force_unsupported: bool = False


def _unsupported_reason(capabilities, language: str) -> str | None:
    """Why this provider should not be asked to transcribe ``language``.

    Capabilities declare this, so the harness can say "English-only" instead of
    scoring a language the model never claimed to support and reporting a word
    error rate near 100%, which would read as a broken model rather than a
    model that was never meant to do it.
    """
    if not language or capabilities.languages is None:
        return None
    if language in capabilities.languages:
        return None
    if len(capabilities.languages) == 1:
        return f"{capabilities.languages[0]} only"
    return "supports " + ", ".join(capabilities.languages)


def evaluate(
    settings: Settings,
    pool: WorkerPool,
    clips: Iterable[Clip],
    options: EvaluationOptions,
    progress: Callable[[str], None] | None = None,
) -> list[ClipResult]:
    """Run every clip through every provider and score the results."""
    clips = list(clips)
    results: list[ClipResult] = []

    for index, clip in enumerate(clips, start=1):
        # A manifest written before this resolved paths may hold relative ones.
        path = Path(clip.path)
        if not path.is_absolute():
            path = settings.root / path
        if progress:
            progress(
                f"[{index}/{len(clips)}] {clip.clip_id} ({clip.language}) "
                f"{(clip.reference[:40] + '...') if len(clip.reference) > 40 else clip.reference}"
            )

        if not path.exists():
            for provider in options.providers:
                results.append(
                    ClipResult(
                        clip_id=clip.clip_id,
                        source=clip.source,
                        language=clip.language,
                        provider=provider,
                        status=ERROR,
                        reference=clip.reference,
                        note=f"audio missing at {path}",
                    )
                )
            continue

        try:
            audio = prepare_audio(path, settings.audio_dir, settings.defaults.target_sample_rate)
        except Exception as exc:
            logger.warning("could not prepare %s: %s", clip.clip_id, exc)
            for provider in options.providers:
                results.append(
                    ClipResult(
                        clip_id=clip.clip_id,
                        source=clip.source,
                        language=clip.language,
                        provider=provider,
                        status=ERROR,
                        reference=clip.reference,
                        note=f"audio could not be decoded: {exc}",
                    )
                )
            continue

        for provider in options.providers:
            config = settings.providers.get(provider)
            if config is None:
                results.append(
                    ClipResult(
                        clip_id=clip.clip_id,
                        source=clip.source,
                        language=clip.language,
                        provider=provider,
                        status=ERROR,
                        reference=clip.reference,
                        note="provider is not configured",
                    )
                )
                continue

            capabilities = capabilities_for(config)
            reason = _unsupported_reason(capabilities, clip.language)
            if reason and not options.force_unsupported:
                results.append(
                    ClipResult(
                        clip_id=clip.clip_id,
                        source=clip.source,
                        language=clip.language,
                        provider=provider,
                        status=UNSUPPORTED,
                        reference=clip.reference,
                        note=reason,
                        audio_duration_ms=audio.duration_ms,
                    )
                )
                continue

            # Only force a language when the provider honours it; Phonon's server
            # accepts the field and ignores it, so sending it would imply a
            # request that is not actually being made.
            language = (
                clip.language
                if capabilities.supports_language_selection
                else None
            )
            transcription_options = TranscriptionOptions(
                language=language, word_timestamps=False
            )

            try:
                result = pool.run(provider, audio, transcription_options)
            except (ProviderError, ProviderUnavailable, ProviderTimeout) as exc:
                results.append(
                    ClipResult(
                        clip_id=clip.clip_id,
                        source=clip.source,
                        language=clip.language,
                        provider=provider,
                        status=ERROR,
                        reference=clip.reference,
                        note=str(exc),
                        audio_duration_ms=audio.duration_ms,
                    )
                )
                continue

            score = metrics.score(clip.reference, result.text, options.policy)
            results.append(
                ClipResult(
                    clip_id=clip.clip_id,
                    source=clip.source,
                    language=clip.language,
                    provider=provider,
                    status=OK,
                    reference=clip.reference,
                    text=result.text,
                    note=(
                        "forced on an unsupported language"
                        if reason and options.force_unsupported
                        else ""
                    ),
                    wer=score.unit_error_rate,
                    cer=score.char_error_rate,
                    reference_words=score.reference_units,
                    word_errors=score.unit_errors,
                    substitutions=score.counts.substitutions,
                    insertions=score.counts.insertions,
                    deletions=score.counts.deletions,
                    reference_chars=score.char_reference_units,
                    char_errors=score.char_errors,
                    latency_ms=result.latency_ms,
                    real_time_factor=result.real_time_factor,
                    audio_duration_ms=result.audio_duration_ms,
                )
            )

    return results
