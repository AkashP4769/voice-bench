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

import numpy as np

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
from app.streaming.events import StreamEvent, StreamMetrics, StreamResult
from app.streaming.segmenter import RollingSegmenter
from app.streaming.session import StreamOptions

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

    # -- streaming ---------------------------------------------------------

    def open_stream(self, options: StreamOptions) -> "PhononStreamSession":
        if self._speech is None:
            self.warmup()
        return PhononStreamSession(self._speech, options)


def _load_live_session():
    """Fermion's live session, or ``None`` if this build does not expose it.

    It lives in a private module (``fermion._speech.live``), so it is treated as
    optional: if it moves or its signature changes, the session falls back to our
    own rolling segmenter and realtime keeps working.
    """
    try:
        from fermion._speech.live import LiveSession  # type: ignore[attr-defined]
    except Exception:
        return None
    return LiveSession


def live_session_available() -> bool:
    """Whether Phonon can stream using Fermion's own segmentation."""
    return _load_live_session() is not None


class PhononStreamSession:
    """Live Phonon transcription.

    Prefers Fermion's ``LiveSession`` — the same segmentation ``fermion listen``
    and Fermion's own WebSocket endpoint use — because it does the work this
    harness would otherwise have to do: RMS voice gating, partials at a
    0.35 s/0.50 s cadence, and closing a segment on 0.70 s of trailing silence.
    It drives the public ``transcribe_array``, so it needs no special engine
    support and no separate server process.

    Word timestamps are not available on this path; the UI gates that option off
    for realtime rather than pretending otherwise.
    """

    SAMPLE_RATE = 16000

    def __init__(self, speech, options: StreamOptions) -> None:
        self._speech = speech
        self._options = options
        self.metrics = StreamMetrics()
        self._finals: list[str] = []
        self._partial = ""
        self._events: list[StreamEvent] = []
        self._error: str | None = None
        self._started = time.perf_counter()
        # LiveSession feeds in 50 ms blocks.
        self._block = int(0.05 * self.SAMPLE_RATE)

        self._live = self._try_live_session(speech)
        if self._live is None:
            self.implementation = "rolling"
            self._segmenter = RollingSegmenter(self._decode, metrics=self.metrics)
        else:
            self.implementation = "live_session"
            self._segmenter = None

    @staticmethod
    def _try_live_session(speech):
        live_cls = _load_live_session()
        if live_cls is None:
            return None
        try:
            return live_cls(speech, on_partial=None, on_final=None)
        except Exception as exc:
            # Signature drift in a private module lands here, not as a crash.
            logger.warning("fermion live session unavailable, rolling instead: %s", exc)
            return None

    def _decode(self, audio) -> str:
        text, _, _ = self._speech.transcribe_array(audio)
        return text or ""

    # -- callbacks from LiveSession ---------------------------------------

    def _on_partial(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._partial = text
        self._events.append(
            StreamEvent.partial(self._best_text(), received_s=self.metrics.audio_s)
        )

    def _on_final(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._finals.append(text)
        self._partial = ""
        self.metrics.segments += 1
        self._events.append(
            StreamEvent.final(
                text,
                segment=len(self._finals),
                received_s=self.metrics.audio_s,
            )
        )

    def _best_text(self) -> str:
        parts = [*self._finals]
        if self._partial:
            parts.append(self._partial)
        return " ".join(part for part in parts if part).strip()

    # -- session contract --------------------------------------------------

    def feed(self, samples) -> list[StreamEvent]:
        array = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        if array.size == 0:
            return []

        self.metrics.chunks += 1
        self.metrics.audio_s += array.size / self.SAMPLE_RATE
        started = time.perf_counter()
        try:
            if self._segmenter is not None:
                events = self._segmenter.feed(array)
            else:
                # LiveSession has no return value: it reports through callbacks.
                for start in range(0, array.size, self._block):
                    self._live.feed(array[start : start + self._block])
                events = list(self._events)
                self._events.clear()
        finally:
            self.metrics.observe_pass((time.perf_counter() - started) * 1000.0)

        if self.metrics.first_partial_ms is None and events:
            self.metrics.first_partial_ms = (
                time.perf_counter() - self._started
            ) * 1000.0
        self.metrics.text_chars = len(self._best_text())
        return events

    def stop(self) -> StreamResult:
        if self._segmenter is not None:
            text, _ = self._segmenter.finish()
            return StreamResult(
                provider="",
                text=text,
                metrics=self.metrics,
                error=self._error,
                implementation="rolling",
            )

        try:
            text = self._live.finish()
        except Exception as exc:
            self._error = f"phonon failed while finishing the stream: {exc}"
            text = self._best_text()
        return StreamResult(
            provider="",
            text=(text or "").strip() or self._best_text(),
            metrics=self.metrics,
            error=self._error,
            implementation="live_session",
        )
