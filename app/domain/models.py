"""Normalized domain types shared by every provider and transport.

The whole point of these types is that the UI, the scoring code and the database
never need to know which engine produced a transcript, or whether it ran in
process or behind HTTP.

Absent values stay ``None``. A provider that cannot measure time-to-first-token
must not report ``0``, because ``0`` is a claim and ``None`` is the truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Transport(str, Enum):
    """How a provider was reached."""

    IN_PROCESS = "in_process"
    OPENAI_HTTP = "openai_http"


class WindowingStrategy(str, Enum):
    """How the harness split audio that exceeded a provider's limit.

    Recorded on every run because our chunking is a confounder for word error
    rate: a transcript produced from chunks reflects the harness as much as the
    model.
    """

    SINGLE_PASS = "single_pass"
    CHUNKED_WORD_DEDUPE = "chunked_word_dedupe"
    CHUNKED_TEXT_CONCAT = "chunked_text_concat"


@dataclass(frozen=True)
class Word:
    """One word with absolute times measured from the start of the audio."""

    text: str
    start_ms: float
    end_ms: float
    probability: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "probability": self.probability,
        }


@dataclass(frozen=True)
class WindowingInfo:
    """How the audio was segmented.

    ``window_count`` is how many pieces *the harness* cut the audio into;
    ``engine_window_count`` is how many pieces *the provider's own engine* used,
    which both of these providers report. They are separate numbers because they
    are separate claims.
    """

    strategy: WindowingStrategy
    window_count: int
    engine_window_count: int | None = None
    window_seconds: float | None = None
    overlap_seconds: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "window_count": self.window_count,
            "engine_window_count": self.engine_window_count,
            "window_seconds": self.window_seconds,
            "overlap_seconds": self.overlap_seconds,
        }


@dataclass(frozen=True)
class PreparedAudio:
    """Audio the harness has normalized and identified.

    ``sha256`` is taken over the normalized PCM samples rather than the source
    file, so the same recording in two different containers hashes identically.
    That hash is the join key that makes two providers comparable and a result
    reproducible.
    """

    path: Path
    sha256: str
    duration_ms: float
    sample_rate: int
    channels: int

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000.0


@dataclass(frozen=True)
class TranscriptionOptions:
    """Per-request provider options.

    Not every provider honours every field; ``ProviderCapabilities`` is what the
    UI and the orchestrator use to decide what may be requested.
    """

    language: str | None = None
    word_timestamps: bool = False
    keywords: tuple[str, ...] = ()
    hotword_lambda: float | None = None

    def fingerprint(self) -> dict[str, Any]:
        """Stable representation for reproducibility records."""
        return {
            "language": self.language,
            "word_timestamps": self.word_timestamps,
            "keywords": list(self.keywords),
            "hotword_lambda": self.hotword_lambda,
        }


@dataclass
class TranscriptionResult:
    """Provider output normalized into one shape.

    Provider-native payloads are kept in ``raw_response`` for debugging, but the
    UI must depend only on the normalized fields.
    """

    provider: str
    model: str
    transport: str
    text: str = ""
    language: str | None = None
    words: list[Word] = field(default_factory=list)
    model_version: str | None = None
    audio_duration_ms: float | None = None
    latency_ms: float | None = None
    time_to_first_token_ms: float | None = None
    real_time_factor: float | None = None
    windowing: WindowingInfo | None = None
    # True when a provider warns that it ran out of token budget, which means
    # part of the audio may be missing from the transcript.
    truncated: bool | None = None
    error: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def warnings(self) -> list[str]:
        """Accuracy caveats a reader must see alongside the numbers."""
        notes: list[str] = []
        if self.truncated:
            notes.append(
                "the provider reported truncated output: part of the audio may be "
                "missing from this transcript"
            )
        if self.windowing and self.windowing.window_count > 1:
            notes.append(
                f"audio was split into {self.windowing.window_count} windows by the "
                "harness, which affects word error rate"
            )
        return notes

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "model_version": self.model_version,
            "transport": self.transport,
            "text": self.text,
            "language": self.language,
            "words": [w.as_dict() for w in self.words],
            "audio_duration_ms": self.audio_duration_ms,
            "latency_ms": self.latency_ms,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "real_time_factor": self.real_time_factor,
            "windowing": self.windowing.as_dict() if self.windowing else None,
            "truncated": self.truncated,
            "warnings": self.warnings,
            "error": self.error,
            "raw_response": self.raw_response,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TranscriptionResult":
        """Rebuild a result that crossed a process boundary as JSON."""
        words = [
            Word(
                text=entry.get("text", ""),
                start_ms=float(entry.get("start_ms") or 0.0),
                end_ms=float(entry.get("end_ms") or 0.0),
                probability=entry.get("probability"),
            )
            for entry in payload.get("words") or []
        ]

        windowing = None
        raw_windowing = payload.get("windowing")
        if raw_windowing:
            windowing = WindowingInfo(
                strategy=WindowingStrategy(
                    raw_windowing.get("strategy", WindowingStrategy.SINGLE_PASS.value)
                ),
                window_count=int(raw_windowing.get("window_count") or 1),
                engine_window_count=raw_windowing.get("engine_window_count"),
                window_seconds=raw_windowing.get("window_seconds"),
                overlap_seconds=raw_windowing.get("overlap_seconds"),
            )

        return cls(
            provider=payload.get("provider", ""),
            model=payload.get("model", ""),
            model_version=payload.get("model_version"),
            transport=payload.get("transport", ""),
            text=payload.get("text") or "",
            language=payload.get("language"),
            words=words,
            audio_duration_ms=payload.get("audio_duration_ms"),
            latency_ms=payload.get("latency_ms"),
            time_to_first_token_ms=payload.get("time_to_first_token_ms"),
            real_time_factor=payload.get("real_time_factor"),
            windowing=windowing,
            truncated=payload.get("truncated"),
            error=payload.get("error"),
            raw_response=payload.get("raw_response") or {},
        )
