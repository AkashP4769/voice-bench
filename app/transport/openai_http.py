"""OpenAI-compatible HTTP transport.

Speaks ``POST {base_url}/v1/audio/transcriptions`` with ``verbose_json`` and
word timestamps. Fermion's ``fermion serve phonon-2`` implements exactly this,
and so do the hosted transcription APIs.

That equivalence is the whole reason this class exists: switching a provider
from local to hosted is a change of ``base_url`` and ``api_key_env`` in
``providers.yaml``, with no new code. The key is read at call time from the
environment and never stored on the object.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx

from app.domain.models import (
    PreparedAudio,
    TranscriptionOptions,
    TranscriptionResult,
    Transport,
    WindowingInfo,
    WindowingStrategy,
    Word,
)
from app.errors import ProviderError, ProviderTimeout
from app.scoring.normalize import language_code

logger = logging.getLogger(__name__)


def resolve_endpoint(base_url: str) -> str:
    """Accept a base URL with or without the ``/v1`` suffix.

    The OpenAI SDKs are configured with ``.../v1``; a bare host is the more
    natural thing to write in a config file. Both should work.
    """
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/audio/transcriptions"
    return f"{base}/v1/audio/transcriptions"


class OpenAiHttpTransport:
    name = Transport.OPENAI_HTTP.value

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        self.base_url = base_url
        self._api_key = api_key
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.endpoint = resolve_endpoint(base_url)

    @property
    def requires_key(self) -> bool:
        return bool(self.api_key_env)

    @property
    def has_key(self) -> bool:
        return bool(self._api_key)

    def _headers(self) -> dict[str, str]:
        if self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}
        return {}

    def transcribe(
        self,
        provider,
        audio: PreparedAudio,
        options: TranscriptionOptions,
    ) -> TranscriptionResult:
        data: dict[str, str] = {
            "model": provider.config.model,
            "response_format": "verbose_json",
        }
        if options.word_timestamps:
            # The OpenAI SDKs send this repeated field name verbatim.
            data["timestamp_granularities[]"] = "word"
        if options.keywords:
            data["hotwords"] = ",".join(options.keywords)
        if options.language:
            data["language"] = options.language

        path = Path(audio.path)
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                with path.open("rb") as handle:
                    response = client.post(
                        self.endpoint,
                        headers=self._headers(),
                        data=data,
                        files={"file": (path.name, handle, "audio/wav")},
                    )
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(
                f"{provider.name} did not respond within {self.timeout_s:g}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"{provider.name} could not reach {self.endpoint}: {exc}. "
                "Start the local server, or check the configured base_url."
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if response.status_code != 200:
            raise ProviderError(self._describe_error(response))

        try:
            payload = response.json()
        except ValueError:
            # response_format=text returns a bare transcript.
            payload = {"text": response.text}

        return self._to_result(provider, audio, payload, elapsed_ms, options)

    def _describe_error(self, response: httpx.Response) -> str:
        detail = response.text.strip()[:300]
        hint = ""
        if response.status_code == 401:
            variable = self.api_key_env or "the configured API key variable"
            hint = (
                f" The endpoint requires a key; {variable} is not set"
                if not self.has_key
                else " The configured API key was rejected"
            )
        elif response.status_code == 404:
            hint = " The endpoint does not serve that model name."
        elif response.status_code == 503:
            hint = " The server queue is full; retry shortly."
        return f"HTTP {response.status_code} from {self.endpoint}: {detail}{hint}"

    def _to_result(
        self,
        provider,
        audio: PreparedAudio,
        payload: dict,
        elapsed_ms: float,
        options: TranscriptionOptions,
    ) -> TranscriptionResult:
        words: list[Word] = []
        for entry in payload.get("words") or []:
            text = entry.get("word") or entry.get("text") or ""
            if not text:
                continue
            words.append(
                Word(
                    text=text,
                    # OpenAI-compatible word times are seconds.
                    start_ms=float(entry.get("start") or 0.0) * 1000.0,
                    end_ms=float(entry.get("end") or 0.0) * 1000.0,
                    probability=entry.get("probability"),
                )
            )

        duration_ms = audio.duration_ms
        reported = payload.get("duration")
        if isinstance(reported, (int, float)) and reported > 0:
            duration_ms = float(reported) * 1000.0

        vendor = payload.get("x_fermion") or {}

        # `x_fermion` is Fermion-specific and absent on other servers, so it is
        # read opportunistically: when present it carries the truncation flag
        # and the engine's own window count, both of which are accuracy facts
        # the generic OpenAI shape does not expose.
        truncated = None
        engine_windows = None
        if isinstance(vendor.get("truncated"), bool):
            truncated = vendor["truncated"]
        if isinstance(vendor.get("window_count"), int):
            engine_windows = vendor["window_count"]

        result = TranscriptionResult(
            provider=provider.name,
            model=provider.config.model,
            transport=self.name,
            text=payload.get("text") or "",
            language=language_code(payload.get("language")),
            words=words,
            model_version=vendor.get("model"),
            audio_duration_ms=duration_ms,
            latency_ms=elapsed_ms,
            # A round trip over HTTP cannot separate queueing from decode, so
            # time-to-first-token is left unmeasured rather than guessed.
            time_to_first_token_ms=None,
            real_time_factor=elapsed_ms / duration_ms if duration_ms > 0 else None,
            windowing=WindowingInfo(
                strategy=WindowingStrategy.SINGLE_PASS,
                window_count=1,
                engine_window_count=engine_windows,
            ),
            truncated=truncated,
            raw_response=payload,
        )
        return result
