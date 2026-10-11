"""Shared provider behaviour.

A provider owns three things: its declared capabilities, the transport it is
reached through, and the engine-specific call that runs locally. Everything
else — capability checks, timing, real-time factor, result shape — is common
and lives here.
"""

from __future__ import annotations

import logging
import time

from app.config import ProviderConfig, Settings
from app.domain.models import (
    PreparedAudio,
    TranscriptionOptions,
    TranscriptionResult,
    WindowingInfo,
    WindowingStrategy,
)
from app.domain.provider import ProviderCapabilities
from app.errors import ProviderError
from app.transport.in_process import InProcessTransport
from app.transport.openai_http import OpenAiHttpTransport

logger = logging.getLogger(__name__)


class BaseProvider:
    # Defaults; an instance takes its name from configuration.
    name: str = "base"
    provider_type: str = "base"
    capabilities: ProviderCapabilities = ProviderCapabilities()

    def __init__(self, config: ProviderConfig, settings: Settings) -> None:
        self.config = config
        self.settings = settings
        # The configured name is the identity that results, API requests and
        # stored runs use. The class attribute is only the adapter type, and
        # one adapter type can be configured more than once (for example a
        # local and a hosted instance of the same model).
        self.name = config.name
        self.provider_type = config.type
        self.transport = self._build_transport()

    def _build_transport(self):
        if self.config.transport == "openai_http":
            server = self.config.server
            if server is None or not server.base_url:
                raise ProviderError(
                    f"provider {self.name!r} uses the openai_http transport "
                    "but has no server.base_url configured"
                )
            return OpenAiHttpTransport(
                base_url=server.base_url,
                api_key=self.config.api_key(),
                api_key_env=server.api_key_env,
                timeout_s=server.timeout_s,
            )
        return InProcessTransport()

    def transcribe(
        self,
        audio: PreparedAudio,
        options: TranscriptionOptions,
    ) -> TranscriptionResult:
        problems = self.capabilities.accepts(options)
        if problems:
            raise ProviderError(
                f"provider {self.name!r} cannot honour the requested options: "
                + "; ".join(problems)
            )
        return self.transport.transcribe(self, audio, options)

    def transcribe_local(
        self,
        audio: PreparedAudio,
        options: TranscriptionOptions,
    ) -> TranscriptionResult:
        """Run the engine in this process. Implemented by each adapter."""
        raise NotImplementedError

    def describe(self) -> dict:
        """Runtime facts for the reproducibility record."""
        return {}

    def _timed(
        self,
        audio: PreparedAudio,
        options: TranscriptionOptions,
        transport_name: str,
    ):
        """Wall-clock timing around a provider call.

        Engine load time is deliberately excluded by the caller: a warm-up pass
        loads the model so the first timed run measures inference, not disk.
        """
        started = time.perf_counter()
        result = self.transcribe_local(audio, options)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        result.provider = self.name
        result.transport = transport_name
        result.model = self.config.model
        result.audio_duration_ms = audio.duration_ms
        result.latency_ms = elapsed_ms
        if result.real_time_factor is None and audio.duration_ms > 0:
            result.real_time_factor = elapsed_ms / audio.duration_ms
        if result.windowing is None:
            result.windowing = WindowingInfo(
                strategy=WindowingStrategy.SINGLE_PASS, window_count=1
            )
        return result
