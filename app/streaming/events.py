"""Streaming events and results.

An event is what a provider reports *while* audio is still arriving. It crosses
two boundaries — worker process to API, then API to browser — so it has a single
flat, JSON-friendly shape rather than a class hierarchy, with constructor helpers
so call sites still read clearly.

One deliberate difference from the batch path: there, a missing measurement is
``None`` and never ``0``. The same rule holds here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StreamKind(str, Enum):
    """What an event is reporting."""

    # The in-flight hypothesis. Replaces the previous partial; it is not a delta.
    PARTIAL = "partial"
    # A segment closed and its text is settled.
    FINAL = "final"
    # Lifecycle: starting, streaming, stopped, error, overrun.
    STATUS = "status"
    # Per-provider timing and counters.
    METRICS = "metrics"
    # Something went wrong for this provider. Other providers keep running.
    ERROR = "error"


class StreamState(str, Enum):
    """Lifecycle states reported through a STATUS event."""

    STARTING = "starting"
    STREAMING = "streaming"
    STOPPED = "stopped"
    ERROR = "error"
    # The provider fell behind and its session was stopped deliberately.
    OVERRUN = "overrun"


@dataclass
class StreamEvent:
    """One thing a provider reported.

    ``provider`` is filled in by the session manager, which owns the routing, so
    adapters do not need to know their configured name.
    """

    kind: StreamKind
    provider: str = ""
    # The current best FULL transcript for the session, for a partial event:
    # everything settled so far plus the in-flight hypothesis. For a final event
    # it is the settled segment's text, with `segment` giving its ordinal. The UI
    # renders `text` directly, so a provider that reports only segment-sized
    # hypotheses is responsible for prepending what it has already settled.
    text: str = ""
    # Text the engine has not yet committed. Only Whistle reports it.
    pending: str | None = None
    # Segment ordinal, when the engine closes segments explicitly.
    segment: int | None = None
    # Seconds of audio the engine has consumed.
    received_s: float | None = None
    # How long the engine's last decode pass took, in milliseconds.
    pass_ms: float | None = None
    state: StreamState | None = None
    detail: str | None = None
    message: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    # -- constructors ------------------------------------------------------

    @classmethod
    def partial(
        cls,
        text: str,
        *,
        pending: str | None = None,
        received_s: float | None = None,
        pass_ms: float | None = None,
    ) -> "StreamEvent":
        return cls(
            kind=StreamKind.PARTIAL,
            text=text,
            pending=pending,
            received_s=received_s,
            pass_ms=pass_ms,
        )

    @classmethod
    def final(
        cls,
        text: str,
        *,
        segment: int | None = None,
        received_s: float | None = None,
        pass_ms: float | None = None,
    ) -> "StreamEvent":
        return cls(
            kind=StreamKind.FINAL,
            text=text,
            segment=segment,
            received_s=received_s,
            pass_ms=pass_ms,
        )

    @classmethod
    def status(
        cls,
        state: StreamState,
        *,
        detail: str | None = None,
    ) -> "StreamEvent":
        return cls(kind=StreamKind.STATUS, state=state, detail=detail)

    @classmethod
    def error(cls, message: str) -> "StreamEvent":
        return cls(kind=StreamKind.ERROR, message=message)

    @classmethod
    def metrics_event(
        cls, metrics: dict[str, Any], *, provider: str = ""
    ) -> "StreamEvent":
        return cls(kind=StreamKind.METRICS, provider=provider, metrics=metrics)

    # -- serialization -----------------------------------------------------

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.kind.value}
        if self.provider:
            payload["provider"] = self.provider
        if self.kind in (StreamKind.PARTIAL,):
            payload["text"] = self.text
            if self.pending is not None:
                payload["pending"] = self.pending
        elif self.kind is StreamKind.FINAL:
            payload["text"] = self.text
            if self.segment is not None:
                payload["segment"] = self.segment
        elif self.kind is StreamKind.STATUS:
            payload["state"] = self.state.value if self.state else None
            if self.detail:
                payload["detail"] = self.detail
        elif self.kind is StreamKind.ERROR:
            payload["message"] = self.message or ""
        elif self.kind is StreamKind.METRICS:
            payload["metrics"] = self.metrics

        if self.received_s is not None:
            payload["receivedS"] = round(self.received_s, 3)
        if self.pass_ms is not None:
            payload["passMs"] = round(self.pass_ms, 1)
        return payload

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> "StreamEvent":
        state = payload.get("state")
        return cls(
            kind=StreamKind(payload["type"]),
            provider=payload.get("provider", ""),
            text=payload.get("text") or "",
            pending=payload.get("pending"),
            segment=payload.get("segment"),
            received_s=payload.get("receivedS"),
            pass_ms=payload.get("passMs"),
            state=StreamState(state) if state else None,
            detail=payload.get("detail"),
            message=payload.get("message"),
            metrics=payload.get("metrics") or {},
        )


@dataclass
class StreamMetrics:
    """Timing and volume for one provider's session.

    ``mean_pass_ms`` against the chunk duration is the honest measure of whether
    an engine is keeping up with realtime. Reporting it is the point; a stream
    that quietly falls behind would otherwise look like a model that stopped
    hearing speech.
    """

    chunks: int = 0
    audio_s: float = 0.0
    decodes: int = 0
    mean_pass_ms: float | None = None
    max_pass_ms: float | None = None
    first_partial_ms: float | None = None
    segments: int = 0
    text_chars: int = 0
    overruns: int = 0

    def observe_pass(self, pass_ms: float) -> None:
        self.decodes += 1
        if self.mean_pass_ms is None:
            self.mean_pass_ms = pass_ms
        else:
            # Running mean, so a long session does not keep every sample.
            self.mean_pass_ms += (pass_ms - self.mean_pass_ms) / self.decodes
        if self.max_pass_ms is None or pass_ms > self.max_pass_ms:
            self.max_pass_ms = pass_ms

    @property
    def realtime_factor(self) -> float | None:
        """Total decode time divided by total audio time.

        Above 1 means the engine spends longer decoding than the audio lasts, so
        it is falling behind. This is the number that makes a lagging stream
        visible instead of looking like a model that stopped hearing speech.
        """
        if self.audio_s <= 0 or self.mean_pass_ms is None or self.decodes == 0:
            return None
        decoded_s = self.mean_pass_ms * self.decodes / 1000.0
        return decoded_s / self.audio_s

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunks": self.chunks,
            "audio_s": round(self.audio_s, 2),
            "decodes": self.decodes,
            "mean_pass_ms": (
                round(self.mean_pass_ms, 1) if self.mean_pass_ms is not None else None
            ),
            "max_pass_ms": (
                round(self.max_pass_ms, 1) if self.max_pass_ms is not None else None
            ),
            "first_partial_ms": (
                round(self.first_partial_ms, 1)
                if self.first_partial_ms is not None
                else None
            ),
            "segments": self.segments,
            "text_chars": self.text_chars,
            "overruns": self.overruns,
            "realtime_factor": (
                round(self.realtime_factor, 3)
                if self.realtime_factor is not None
                else None
            ),
        }


@dataclass
class StreamResult:
    """What a finished session produced."""

    provider: str
    text: str = ""
    words: list[dict[str, Any]] = field(default_factory=list)
    metrics: StreamMetrics = field(default_factory=StreamMetrics)
    error: str | None = None
    # Which implementation served it: "live_session", "rolling" or "native".
    implementation: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "text": self.text,
            "words": self.words,
            "metrics": self.metrics.as_dict(),
            "error": self.error,
            "implementation": self.implementation,
        }
