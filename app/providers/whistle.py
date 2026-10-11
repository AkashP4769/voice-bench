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
"""

from __future__ import annotations

import logging
import os
import time

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
        self._version: str | None = None
        self._load_seconds: float | None = None

    def warmup(self) -> float:
        """Load the engine and weights so the first timed run measures inference.

        The package fetches a platform engine and ``whistle.cact`` on first use,
        which takes far longer than a decode. Timing that as latency would make
        every first run meaningless.
        """
        started = time.perf_counter()
        needle = _import_needle()
        self._needle = needle
        self._version = getattr(needle, "__version__", None)
        try:
            needle.Whistle()
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
