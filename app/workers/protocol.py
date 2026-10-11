"""Wire protocol between the API process and a provider worker.

Line-delimited JSON over stdin/stdout: one request line in, one response line
out. Text is deliberately avoided in favour of JSON so that audio metadata,
options and results survive the boundary without ad-hoc parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.domain.models import PreparedAudio, TranscriptionOptions

READY = "ready"
FATAL = "fatal"
JOB = "job"
RESULT = "result"
SHUTDOWN = "shutdown"

# Streaming frames. A session is driven by three client messages and answered by
# one reply each, which keeps the one-response-per-request invariant the pool
# relies on for id correlation and timeouts.
STREAM_START = "stream_start"
STREAM_STARTED = "stream_started"
STREAM_CHUNK = "stream_chunk"
STREAM_EVENTS = "stream_events"
STREAM_STOP = "stream_stop"
STREAM_DONE = "stream_done"
STREAM_ERROR = "stream_error"


def encode_audio(audio: PreparedAudio) -> dict[str, Any]:
    return {
        "path": str(audio.path),
        "sha256": audio.sha256,
        "duration_ms": audio.duration_ms,
        "sample_rate": audio.sample_rate,
        "channels": audio.channels,
    }


def decode_audio(payload: dict[str, Any]) -> PreparedAudio:
    return PreparedAudio(
        path=payload["path"],
        sha256=payload["sha256"],
        duration_ms=payload["duration_ms"],
        sample_rate=payload["sample_rate"],
        channels=payload["channels"],
    )


def encode_options(options: TranscriptionOptions) -> dict[str, Any]:
    return {
        "language": options.language,
        "word_timestamps": options.word_timestamps,
        "keywords": list(options.keywords),
        "hotword_lambda": options.hotword_lambda,
    }


def decode_options(payload: dict[str, Any]) -> TranscriptionOptions:
    return TranscriptionOptions(
        language=payload.get("language"),
        word_timestamps=bool(payload.get("word_timestamps")),
        keywords=tuple(payload.get("keywords") or ()),
        hotword_lambda=payload.get("hotword_lambda"),
    )


# -- streaming audio -------------------------------------------------------


def encode_audio_block(samples) -> str:
    """Encode 16 kHz mono float32 samples for the worker channel.

    Base64 inside the JSON line costs about a third extra on a hop that is
    loopback, and it preserves the single framing mechanism this protocol already
    uses. A raw length-prefixed frame would be the optimisation if this ever has
    to cross a real network.
    """
    import base64

    import numpy as np

    array = np.ascontiguousarray(samples, dtype="<f4")
    return base64.b64encode(array.tobytes()).decode("ascii")


def decode_audio_block(encoded: str):
    """Decode a block produced by :func:`encode_audio_block`."""
    import base64

    import numpy as np

    raw = base64.b64decode(encoded)
    return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)


@dataclass(frozen=True)
class JobRequest:
    job_id: str
    audio: PreparedAudio
    options: TranscriptionOptions

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": JOB,
            "id": self.job_id,
            "audio": encode_audio(self.audio),
            "options": encode_options(self.options),
        }

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> "JobRequest":
        return cls(
            job_id=payload["id"],
            audio=decode_audio(payload["audio"]),
            options=decode_options(payload.get("options") or {}),
        )
