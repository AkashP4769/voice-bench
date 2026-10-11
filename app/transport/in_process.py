"""In-process transport.

The fastest and most faithful path: the engine runs in this process, so no audio
is copied and no HTTP layer sits between the harness and the model. Timing
measured here is the model's own, which is why the transport is recorded on
every result — an in-process latency and an HTTP latency are not the same
measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.models import (
    PreparedAudio,
    TranscriptionOptions,
    TranscriptionResult,
    Transport,
)


@dataclass
class InProcessTransport:
    name: str = Transport.IN_PROCESS.value

    def transcribe(
        self,
        provider,
        audio: PreparedAudio,
        options: TranscriptionOptions,
    ) -> TranscriptionResult:
        return provider._timed(audio, options, self.name)
