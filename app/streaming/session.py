"""The streaming session contract.

A session is one provider's live decode. Adapters implement it however their
engine allows — Whistle drives a generator, Phonon rolls partials, and the
fallback re-decodes a window — but the session manager only ever sees these three
methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from app.streaming.events import StreamEvent, StreamResult


@dataclass(frozen=True)
class StreamOptions:
    """Per-session options, gated by declared provider capabilities."""

    language: str | None = None
    keywords: tuple[str, ...] = ()
    word_timestamps: bool = False
    #: Seconds of audio an engine should be handed at a time. Engines that
    #: segment internally (Phonon) ignore it; Whistle needs about 2 s on CPU to
    #: keep up with realtime, so this is a real tuning knob, not a detail.
    chunk_seconds: float | None = None

    def as_wire(self) -> dict:
        return {
            "language": self.language,
            "keywords": list(self.keywords),
            "word_timestamps": self.word_timestamps,
            "chunk_seconds": self.chunk_seconds,
        }

    @classmethod
    def from_wire(cls, payload: dict) -> "StreamOptions":
        chunk = payload.get("chunk_seconds")
        return cls(
            language=payload.get("language"),
            keywords=tuple(payload.get("keywords") or ()),
            word_timestamps=bool(payload.get("word_timestamps")),
            chunk_seconds=float(chunk) if isinstance(chunk, (int, float)) else None,
        )


@runtime_checkable
class StreamingSession(Protocol):
    """A live decode for one provider."""

    #: Which implementation is serving this session, shown in the UI so a
    #: fallback is visible rather than silent.
    implementation: str

    def feed(self, samples: np.ndarray) -> list[StreamEvent]:
        """Accept 16 kHz mono float32 audio and return any events it produced."""
        ...

    def stop(self) -> StreamResult:
        """Finalize and return what the session produced."""
        ...
