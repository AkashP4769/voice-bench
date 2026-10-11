"""Domain types and contracts."""

from app.domain.models import (
    PreparedAudio,
    TranscriptionOptions,
    TranscriptionResult,
    Transport,
    WindowingInfo,
    WindowingStrategy,
    Word,
)
from app.domain.provider import ProviderCapabilities, SpeechProvider

__all__ = [
    "PreparedAudio",
    "ProviderCapabilities",
    "SpeechProvider",
    "TranscriptionOptions",
    "TranscriptionResult",
    "Transport",
    "WindowingInfo",
    "WindowingStrategy",
    "Word",
]
