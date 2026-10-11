"""Transport contract.

A transport is *how* a provider is reached. Keeping this separate from the
adapter is what lets a provider move from in-process to a local HTTP server to a
hosted endpoint without the adapter, the API, the UI or the database changing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain.models import (
    PreparedAudio,
    TranscriptionOptions,
    TranscriptionResult,
)


@runtime_checkable
class BaseTransport(Protocol):
    name: str

    def transcribe(
        self,
        provider,
        audio: PreparedAudio,
        options: TranscriptionOptions,
    ) -> TranscriptionResult: ...
