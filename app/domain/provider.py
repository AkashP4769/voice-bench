"""The provider contract.

Every model behind this interface is interchangeable. Adding a model means
adding one adapter plus one entry in ``providers.yaml``; nothing in the API, the
UI, the scoring code or the storage layer changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.domain.models import PreparedAudio, TranscriptionOptions, TranscriptionResult


@dataclass(frozen=True)
class ProviderCapabilities:
    """What a provider can actually do.

    These are declared, not discovered, and they gate the UI. Two real limits
    this exists to encode: Phonon-2 is English-only and cannot select a
    language, and it accepts at most 25 hotwords.
    """

    # None means multilingual with auto-detection.
    languages: tuple[str, ...] | None = None
    word_timestamps: bool = False
    hotwords: bool = False
    # None means no documented ceiling, matching the vendor's own limit.
    max_hotwords: int | None = None
    embeddings: bool = False
    streaming: bool = False
    # None means the provider segments long audio itself.
    max_duration_s: float | None = None
    # None means the provider ignores language selection.
    supports_language_selection: bool = False

    def as_dict(self) -> dict:
        return {
            "languages": list(self.languages) if self.languages else None,
            "word_timestamps": self.word_timestamps,
            "hotwords": self.hotwords,
            "max_hotwords": self.max_hotwords,
            "embeddings": self.embeddings,
            "streaming": self.streaming,
            "max_duration_s": self.max_duration_s,
            "supports_language_selection": self.supports_language_selection,
        }

    def accepts(self, options: TranscriptionOptions) -> list[str]:
        """Return the reasons ``options`` cannot be honoured, if any."""
        problems: list[str] = []
        if options.language and not self.supports_language_selection:
            problems.append("language selection is not supported")
        if (
            options.language
            and self.languages
            and options.language not in self.languages
        ):
            problems.append(
                f"language {options.language!r} is not supported "
                f"(available: {', '.join(self.languages)})"
            )
        if options.word_timestamps and not self.word_timestamps:
            problems.append("word timestamps are not supported")
        if options.keywords and not self.hotwords:
            problems.append("keyword biasing is not supported")
        if (
            options.keywords
            and self.max_hotwords is not None
            and len(options.keywords) > self.max_hotwords
        ):
            problems.append(
                f"at most {self.max_hotwords} keywords are supported, "
                f"got {len(options.keywords)}"
            )
        return problems


@runtime_checkable
class SpeechProvider(Protocol):
    """A speech-to-text model, reachable however the adapter chooses."""

    name: str
    capabilities: ProviderCapabilities

    def transcribe(
        self,
        audio: PreparedAudio,
        options: TranscriptionOptions,
    ) -> TranscriptionResult: ...

    def describe(self) -> dict:
        """Runtime facts for the reproducibility record, such as CPU tier."""
        ...
