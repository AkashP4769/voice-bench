"""Phonon-2 adapter (Fermion Research, via ``fermion-research``).

Facts this adapter encodes, verified by loading the engine and transcribing:

* ``fermion.load_speech("phonon-2")`` returns an engine object with
  ``transcribe`` (text, decode_seconds, duration) and ``transcribe_detailed``,
  which returns a ``Transcript`` carrying ``text``, ``words``, ``segments``,
  ``truncated``, ``decode_seconds`` and the engine's own ``windows``.
* Word entries use the key ``text`` with ``start``/``end`` in **seconds**.
* The engine segments long audio itself: up to 35 s in one call, then 25-35 s
  windows cut at pauses, so ``max_duration_s`` is ``None``.
* ``truncated`` means a window used its whole token budget and audio may be
  missing, which is an accuracy caveat, not a detail.
* The engine is English-only, and its HTTP server accepts a ``language`` field
  but does not implement it, so language selection is not offered.
* Torch is a heavy dependency here and a Windows install may need
  ``vc_redist.x64.exe``, so the import is deferred and the error is actionable.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from app.domain.models import (
    PreparedAudio,
    TranscriptionOptions,
    TranscriptionResult,
    WindowingInfo,
    WindowingStrategy,
    Word,
)
from app.domain.provider import ProviderCapabilities
from app.errors import ProviderError, ProviderUnavailable
from app.providers.base import BaseProvider

logger = logging.getLogger(__name__)

MAX_HOTWORDS = 25

CAPABILITIES = ProviderCapabilities(
    languages=("en",),
    word_timestamps=True,
    hotwords=True,
    max_hotwords=MAX_HOTWORDS,
    embeddings=False,
    streaming=False,
    # The engine cuts long recordings into pause-aligned windows itself.
    max_duration_s=None,
    # The server accepts `language` but does not implement it.
    supports_language_selection=False,
)

INSTALL_HINT = 'pip install "voice-bench[phonon]"'


def _import_fermion():
    try:
        import fermion  # noqa: PLC0415 - deliberately deferred, pulls in torch
    except ImportError as exc:
        message = (
            f"the phonon provider needs the fermion-research package, which is not "
            f"installed. Install it with: {INSTALL_HINT}"
        )
        if os.name == "nt" and "torch" in str(exc):
            message += (
                ". On Windows, importing torch can fail with WinError 126; install "
                "Microsoft's vc_redist.x64.exe and retry"
            )
        raise ProviderUnavailable(message) from exc
    return fermion


class PhononProvider(BaseProvider):
    name = "phonon"
    capabilities = CAPABILITIES

    def __init__(self, config, settings) -> None:
        super().__init__(config, settings)
        self._speech = None
        self._version: str | None = None
        self._load_seconds: float | None = None

    def warmup(self) -> float:
        """Resolve, download if needed, and load the model.

        The first run in a new environment also compiles the runtime. None of
        that is inference, so it is measured here and excluded from latency.
        """
        started = time.perf_counter()
        fermion = _import_fermion()
        self._version = getattr(fermion, "VERSION", None) or getattr(
            fermion, "__version__", None
        )
        try:
            self._speech = fermion.load_speech(self.config.model)
        except Exception as exc:
            raise ProviderUnavailable(
                f"phonon model {self.config.model!r} could not be loaded: {exc}"
            ) from exc
        self._load_seconds = time.perf_counter() - started
        return self._load_seconds

    def describe(self) -> dict:
        return {
            "package": "fermion-research",
            "package_version": self._version,
            "load_seconds": self._load_seconds,
            "model": self.config.model,
        }

    def _call_kwargs(self, options: TranscriptionOptions) -> dict:
        kwargs: dict = {}
        if options.keywords:
            kwargs["hotwords"] = list(options.keywords)
        if options.hotword_lambda is not None:
            kwargs["hotword_lambda"] = options.hotword_lambda
        return kwargs

    def transcribe_local(
        self,
        audio: PreparedAudio,
        options: TranscriptionOptions,
    ) -> TranscriptionResult:
        if self._speech is None:
            self.warmup()
        speech = self._speech
        # The normalized cache file is exactly the audio that was hashed and
        # scored, so the engine reads the same bytes every provider sees.
        path = str(Path(audio.path))
        kwargs = self._call_kwargs(options)

        try:
            if options.word_timestamps:
                transcript = speech.transcribe_detailed(path, **kwargs)
            else:
                transcript = None
                text, decode_seconds, audio_seconds = speech.transcribe(path, **kwargs)
        except Exception as exc:
            raise ProviderError(f"phonon failed to transcribe: {exc}") from exc

        if transcript is not None:
            return self._from_transcript(transcript)
        return self._from_plain(text, decode_seconds, audio_seconds)

    def _from_transcript(self, transcript) -> TranscriptionResult:
        words: list[Word] = []
        for entry in getattr(transcript, "words", None) or []:
            text = entry.get("text") or ""
            if not text:
                continue
            words.append(
                Word(
                    text=text,
                    # Phonon reports word times in seconds.
                    start_ms=float(entry.get("start") or 0.0) * 1000.0,
                    end_ms=float(entry.get("end") or 0.0) * 1000.0,
                    # The decoder emits timings, not per-word confidences.
                    probability=None,
                )
            )

        windows = getattr(transcript, "windows", None) or []
        segments = getattr(transcript, "segments", None) or []
        return TranscriptionResult(
            provider=self.name,
            model=self.config.model,
            model_version=self._version,
            transport=self.transport.name,
            text=getattr(transcript, "text", "") or "",
            # Phonon-2 is English-only and does not detect a language, so this
            # is the model's documented envelope rather than a measurement.
            language="en",
            words=words,
            time_to_first_token_ms=None,
            truncated=bool(getattr(transcript, "truncated", False)),
            windowing=WindowingInfo(
                strategy=WindowingStrategy.SINGLE_PASS,
                window_count=1,
                engine_window_count=len(windows) or None,
            ),
            raw_response={
                "decode_seconds": getattr(transcript, "decode_seconds", None),
                "audio_seconds": getattr(transcript, "audio_seconds", None),
                "segment_count": len(segments),
                "engine_window_count": len(windows) or None,
                "generation_tokens": getattr(transcript, "generation_tokens", None),
                "timed": getattr(transcript, "timed", None),
                "truncated": getattr(transcript, "truncated", None),
            },
        )

    def _from_plain(
        self, text: str, decode_seconds: float, audio_seconds: float
    ) -> TranscriptionResult:
        return TranscriptionResult(
            provider=self.name,
            model=self.config.model,
            model_version=self._version,
            transport=self.transport.name,
            text=text or "",
            language="en",
            words=[],
            time_to_first_token_ms=None,
            # The plain call reports no truncation flag, so this is unknown
            # rather than false.
            truncated=None,
            windowing=WindowingInfo(
                strategy=WindowingStrategy.SINGLE_PASS, window_count=1
            ),
            raw_response={
                "decode_seconds": decode_seconds,
                "audio_seconds": audio_seconds,
            },
        )
