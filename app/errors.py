"""Exception types shared across the harness.

Provider failures are expected, not exceptional: an engine can be uninstalled,
a server can be down, a model can be missing. Each of these carries a message
that says what to do about it.
"""

from __future__ import annotations


class VoiceBenchError(Exception):
    """Base class for harness errors."""


class AudioError(VoiceBenchError):
    """Audio could not be decoded or normalized."""


class ProviderError(VoiceBenchError):
    """A provider failed while transcribing."""


class ProviderUnavailable(ProviderError):
    """A provider's runtime or weights are not usable on this machine."""


class ProviderTimeout(ProviderError):
    """A provider exceeded its configured time limit."""
