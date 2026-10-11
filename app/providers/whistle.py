"""Whistle adapter (Cactus Compute, via ``cactus-needle``).

Facts this adapter encodes, all verified against the installed package and by
running the engine:

* Whistle accepts 16 kHz mono float samples of **any length**; the engine
  windows internally. A 47 s clip came back complete (126/126 words), so
  ``max_duration_s`` is ``None`` and the harness does not chunk it.
* Words come back under the key ``word`` with ``start``/``end`` in **seconds**.
* ``ttft_ms`` is measured by the engine; ``decode_tps`` is decoder throughput.
* One model per process, not thread-safe, so a worker process owns the model.
* The package sends anonymous usage counts unless ``NEEDLE_TELEMETRY=0``.
* ``Whistle.stream(chunks)`` is a real streaming API: feed ~1 s of audio and it
  yields the text it has **committed** (what two decode passes agree on), the
  unconfirmed ``pending`` tail, ``received`` seconds and ``pass_ms``.
"""

from __future__ import annotations

import logging
import os
import queue
import time

import numpy as np

from app.audio.prepare import load_samples
from app.domain.models import (
    PreparedAudio,
    TranscriptionOptions,
    TranscriptionResult,
    Word,
)
from app.domain.provider import ProviderCapabilities
from app.errors import ProviderError, ProviderUnavailable
from app.providers.base import BaseProvider
from app.streaming.events import StreamEvent, StreamMetrics, StreamResult
from app.streaming.session import StreamOptions

logger = logging.getLogger(__name__)

CAPABILITIES = ProviderCapabilities(
    languages=("en", "de", "fr", "es", "it", "nl", "pl"),
    word_timestamps=True,
    hotwords=True,
    embeddings=True,
    streaming=True,
    # The engine segments long audio itself.
    max_duration_s=None,
    supports_language_selection=True,
)

INSTALL_HINT = 'pip install "voice-bench[whistle]"'


def _import_needle():
    """Import cactus-needle, with telemetry off and a usable error if absent."""
    # A local benchmark harness should not report the user's usage anywhere.
    os.environ.setdefault("NEEDLE_TELEMETRY", "0")
    try:
        import needle  # noqa: PLC0415 - deliberately deferred
    except ImportError as exc:
        raise ProviderUnavailable(
            f"the whistle provider needs the cactus-needle package, which is not "
            f"installed. Install it with: {INSTALL_HINT}"
        ) from exc
    return needle


class WhistleProvider(BaseProvider):
    name = "whistle"
    capabilities = CAPABILITIES

    def __init__(self, config, settings) -> None:
        super().__init__(config, settings)
        self._needle = None
        self._whistle = None
        self._version: str | None = None
        self._load_seconds: float | None = None

    def warmup(self) -> float:
        """Load the engine and weights so the first timed run measures inference.

        The package fetches a platform engine and ``whistle.cact`` on first use,
        which takes far longer than a decode. Timing that as latency would make
        every first run meaningless. The instance is kept, because the streaming
        API is a method on it and building a second one would load the weights
        again.
        """
        started = time.perf_counter()
        needle = _import_needle()
        self._needle = needle
        self._version = getattr(needle, "__version__", None)
        try:
            self._whistle = needle.Whistle()
        except Exception as exc:  # engine fetch or init failure
            raise ProviderUnavailable(
                f"whistle could not initialize: {exc}"
            ) from exc
        self._load_seconds = time.perf_counter() - started
        return self._load_seconds

    def describe(self) -> dict:
        return {
            "package": "cactus-needle",
            "package_version": self._version,
            "load_seconds": self._load_seconds,
            "telemetry": os.environ.get("NEEDLE_TELEMETRY", "unset"),
        }

    def transcribe_local(
        self,
        audio: PreparedAudio,
        options: TranscriptionOptions,
    ) -> TranscriptionResult:
        needle = self._needle or _import_needle()
        samples = load_samples(audio.path, audio.sample_rate)
        keywords = list(options.keywords) or None
        try:
            payload = needle.transcribe(
                samples,
                language=options.language,
                keywords=keywords,
                word_timestamps=options.word_timestamps,
            )
        except Exception as exc:
            raise ProviderError(f"whistle failed to transcribe: {exc}") from exc
        return self._to_result(payload)

    def _to_result(self, payload: dict) -> TranscriptionResult:
        words: list[Word] = []
        for entry in payload.get("words") or []:
            text = entry.get("word") or entry.get("text") or ""
            if not text:
                continue
            words.append(
                Word(
                    text=text,
                    # Whistle reports word times in seconds.
                    start_ms=float(entry.get("start") or 0.0) * 1000.0,
                    end_ms=float(entry.get("end") or 0.0) * 1000.0,
                    probability=entry.get("probability"),
                )
            )

        ttft = payload.get("ttft_ms")
        return TranscriptionResult(
            provider=self.name,
            model=self.config.model,
            model_version=self._version,
            transport=self.transport.name,
            text=payload.get("text") or "",
            language=payload.get("language") or None,
            words=words,
            time_to_first_token_ms=float(ttft) if isinstance(ttft, (int, float)) else None,
            raw_response={
                "text": payload.get("text"),
                "language": payload.get("language"),
                "ttft_ms": ttft,
                "decode_tps": payload.get("decode_tps"),
            },
        )

    # -- streaming ---------------------------------------------------------

    def open_stream(self, options: StreamOptions) -> "WhistleStreamSession":
        needle = self._needle or _import_needle()
        if self._whistle is None:
            try:
                self._whistle = needle.Whistle()
            except Exception as exc:
                raise ProviderUnavailable(
                    f"whistle could not initialize: {exc}"
                ) from exc
        return WhistleStreamSession(self._whistle, options)


class WhistleStreamSession:
    """Live transcription using the engine's own streaming API.

    ``Whistle.stream`` is a generator, so it is driven one chunk at a time: a
    chunk goes into an inbox, and ``next()`` returns what that chunk produced.
    That keeps the engine's internal commit logic — it commits the words two
    decode passes agree on — and avoids us re-implementing segmentation.
    """

    implementation = "native"

    def __init__(self, whistle, options: StreamOptions) -> None:
        self._options = options
        self.metrics = StreamMetrics()
        self._committed = ""
        self._words: list[dict] = []
        self._started = time.perf_counter()
        self._inbox: queue.Queue = queue.Queue()
        self._closed = False
        self._error: str | None = None
        self._sample_rate = 16000
        # Audio is accumulated to the requested cadence before being handed to
        # the engine. Feeding 1 s at a time measured at ~1.07x realtime on this
        # CPU, which means falling behind forever; at 2 s it measured ~0.63x and
        # keeps up. So this is the knob that decides whether realtime works.
        self._chunk_seconds = options.chunk_seconds or 2.0
        self._chunk_samples = int(self._chunk_seconds * self._sample_rate)
        self._accumulated: list[np.ndarray] = []
        self._accumulated_samples = 0
        try:
            self._gen = whistle.stream(
                self._source(),
                language=options.language,
                keywords=list(options.keywords) or None,
            )
        except Exception as exc:
            raise ProviderError(f"whistle could not start streaming: {exc}") from exc

    def _source(self):
        """Feeds the engine one chunk per feed() call, then stops."""
        while True:
            item = self._inbox.get()
            if item is None:
                return
            yield item

    def feed(self, samples) -> list[StreamEvent]:
        if self._closed:
            return []
        array = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        if array.size == 0:
            return []

        self.metrics.chunks += 1
        self.metrics.audio_s += array.size / self._sample_rate
        self._accumulated.append(array)
        self._accumulated_samples += array.size

        if self._accumulated_samples < self._chunk_samples:
            # Not enough audio for one engine pass yet. The meter still moves,
            # because chunks and audio seconds were counted above.
            return []

        block = (
            self._accumulated[0]
            if len(self._accumulated) == 1
            else np.concatenate(self._accumulated)
        )
        self._accumulated = []
        self._accumulated_samples = 0
        self._inbox.put(block)
        try:
            result = next(self._gen)
        except StopIteration:
            # The generator is spent; stop() will find nothing left to flush.
            return []
        except Exception as exc:
            raise ProviderError(f"whistle streaming failed: {exc}") from exc
        return self._observe(result)

    def _observe(self, result: dict) -> list[StreamEvent]:
        pass_ms = result.get("pass_ms")
        has_pass = isinstance(pass_ms, (int, float))
        if has_pass:
            self.metrics.observe_pass(float(pass_ms))

        text = (result.get("text") or "").strip()
        pending = (result.get("pending") or "").strip()
        if text:
            self._committed = f"{self._committed} {text}".strip()

        for entry in result.get("words") or []:
            word_text = entry.get("word") or entry.get("text") or ""
            if not word_text:
                continue
            self._words.append(
                {
                    "text": word_text,
                    # Whistle reports word times in seconds from stream start.
                    "start_ms": float(entry.get("start") or 0.0) * 1000.0,
                    "end_ms": float(entry.get("end") or 0.0) * 1000.0,
                    "probability": entry.get("probability"),
                }
            )

        self.metrics.text_chars = len(self._committed)
        if self.metrics.first_partial_ms is None and (self._committed or pending):
            self.metrics.first_partial_ms = (
                time.perf_counter() - self._started
            ) * 1000.0

        received = result.get("received")
        return [
            StreamEvent.partial(
                self._committed,
                pending=pending or None,
                received_s=(
                    float(received)
                    if isinstance(received, (int, float))
                    else self.metrics.audio_s
                ),
                pass_ms=float(pass_ms) if has_pass else None,
            )
        ]

    def stop(self) -> StreamResult:
        if self._closed:
            return self._result()
        self._closed = True

        # Push whatever had not yet reached the cadence, so the tail of a short
        # recording is transcribed rather than discarded.
        if self._accumulated_samples:
            block = (
                self._accumulated[0]
                if len(self._accumulated) == 1
                else np.concatenate(self._accumulated)
            )
            self._accumulated = []
            self._accumulated_samples = 0
            self._inbox.put(block)
            try:
                self._observe(next(self._gen))
            except StopIteration:
                pass
            except Exception as exc:
                self._error = f"whistle failed on the final chunk: {exc}"

        # Ending the source makes the generator flush its tail and stop the
        # engine's stream state.
        self._inbox.put(None)
        try:
            final = next(self._gen)
            self._observe(final)
        except StopIteration:
            pass
        except Exception as exc:
            self._error = f"whistle failed while finishing the stream: {exc}"
        try:
            self._gen.close()
        except Exception:
            # The generator's own except-branch already stopped the engine.
            logger.debug("whistle stream generator did not close cleanly")
        return self._result()

    def _result(self) -> StreamResult:
        return StreamResult(
            provider="",
            text=self._committed,
            words=self._words,
            metrics=self.metrics,
            error=self._error,
            implementation=self.implementation,
        )
