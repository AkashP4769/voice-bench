"""The OpenAI-compatible transport.

This is the seam that makes a hosted provider a configuration change rather
than a code change, so its request shaping and response parsing are tested
without a server.
"""

from __future__ import annotations

import httpx
import pytest

from app.audio.prepare import prepare_audio
from app.domain.models import PreparedAudio, TranscriptionOptions
from app.errors import ProviderError
from app.transport.openai_http import OpenAiHttpTransport, resolve_endpoint


class StubConfig:
    def __init__(self, model: str = "phonon-2") -> None:
        self.model = model


class StubProvider:
    def __init__(self, name: str = "phonon-2", model: str = "phonon-2") -> None:
        self.name = name
        self.config = StubConfig(model)


VERBOSE_JSON = {
    "text": "The transcript, word by word.",
    "task": "transcribe",
    "language": "english",
    "duration": 4.2,
    "segments": [{"id": 0, "start": 0.32, "end": 1.84, "text": "The transcript."}],
    "words": [
        {"word": "The", "start": 0.32, "end": 0.48},
        {"word": "transcript,", "start": 0.48, "end": 1.04},
    ],
    "x_fermion": {
        "model": "FermionResearch/Phonon-2",
        "truncated": False,
        "window_count": 3,
    },
}


def test_endpoint_accepts_base_url_with_or_without_v1():
    # The OpenAI SDKs are configured with .../v1; a bare host is what a config
    # file naturally holds. Both must work.
    assert (
        resolve_endpoint("http://127.0.0.1:8000")
        == "http://127.0.0.1:8000/v1/audio/transcriptions"
    )
    assert (
        resolve_endpoint("http://127.0.0.1:8000/")
        == "http://127.0.0.1:8000/v1/audio/transcriptions"
    )
    assert (
        resolve_endpoint("http://127.0.0.1:8000/v1")
        == "http://127.0.0.1:8000/v1/audio/transcriptions"
    )
    assert (
        resolve_endpoint("https://api.example.com/v1/")
        == "https://api.example.com/v1/audio/transcriptions"
    )


def test_vendor_language_name_is_normalized():
    """Fermion reports "english"; Whistle reports "en"; they must compare equal."""
    transport = OpenAiHttpTransport("http://127.0.0.1:8000")
    provider = StubProvider()
    audio = PreparedAudio(
        path="clip.wav", sha256="x", duration_ms=4200.0, sample_rate=16000, channels=1
    )
    result = transport._to_result(
        provider, audio, VERBOSE_JSON, 250.0, TranscriptionOptions()
    )
    assert result.language == "en"


def test_response_seconds_become_milliseconds():
    transport = OpenAiHttpTransport("http://127.0.0.1:8000")
    result = transport._to_result(
        StubProvider(),
        PreparedAudio("clip.wav", "x", 4200.0, 16000, 1),
        VERBOSE_JSON,
        250.0,
        TranscriptionOptions(),
    )
    assert [word.text for word in result.words] == ["The", "transcript,"]
    assert result.words[0].start_ms == 320.0
    assert result.words[0].end_ms == 480.0
    # HTTP returns no per-word probability, so this stays unknown.
    assert result.words[0].probability is None


def test_vendor_block_supplies_model_version_truncation_and_engine_windows():
    transport = OpenAiHttpTransport("http://127.0.0.1:8000")
    result = transport._to_result(
        StubProvider(),
        PreparedAudio("clip.wav", "x", 4200.0, 16000, 1),
        VERBOSE_JSON,
        250.0,
        TranscriptionOptions(),
    )
    assert result.model_version == "FermionResearch/Phonon-2"
    assert result.truncated is False
    assert result.windowing.engine_window_count == 3


def test_reported_duration_wins_over_local_estimate():
    transport = OpenAiHttpTransport("http://127.0.0.1:8000")
    result = transport._to_result(
        StubProvider(),
        PreparedAudio("clip.wav", "x", 9999.0, 16000, 1),
        VERBOSE_JSON,
        250.0,
        TranscriptionOptions(),
    )
    assert result.audio_duration_ms == pytest.approx(4200.0)


def test_time_to_first_token_is_never_invented_for_http():
    """A round trip cannot separate queueing from decode."""
    transport = OpenAiHttpTransport("http://127.0.0.1:8000")
    result = transport._to_result(
        StubProvider(),
        PreparedAudio("clip.wav", "x", 4200.0, 16000, 1),
        VERBOSE_JSON,
        250.0,
        TranscriptionOptions(),
    )
    assert result.time_to_first_token_ms is None
    assert result.latency_ms == 250.0
    assert result.transport == "openai_http"


def test_plain_text_response_is_accepted():
    """response_format=text returns a bare transcript, not JSON."""
    transport = OpenAiHttpTransport("http://127.0.0.1:8000")
    result = transport._to_result(
        StubProvider(),
        PreparedAudio("clip.wav", "x", 1000.0, 16000, 1),
        {"text": "bare transcript\n"},
        100.0,
        TranscriptionOptions(),
    )
    assert result.text == "bare transcript\n"
    assert result.words == []


def test_missing_vendor_block_is_tolerated():
    transport = OpenAiHttpTransport("http://127.0.0.1:8000")
    payload = {"text": "hello", "language": "en", "duration": 1.0, "words": []}
    result = transport._to_result(
        StubProvider(),
        PreparedAudio("clip.wav", "x", 1000.0, 16000, 1),
        payload,
        50.0,
        TranscriptionOptions(),
    )
    assert result.model_version is None
    assert result.truncated is None
    assert result.windowing.engine_window_count is None


def test_credentials_are_reported_when_configured():
    transport = OpenAiHttpTransport(
        "https://api.example.com/v1", api_key="secret", api_key_env="MY_KEY"
    )
    assert transport.has_key is True
    assert transport.requires_key is True
    assert transport._headers() == {"Authorization": "Bearer secret"}


def test_error_message_names_the_variable_but_not_the_value():
    transport = OpenAiHttpTransport(
        "https://api.example.com/v1", api_key=None, api_key_env="MY_KEY"
    )
    response = httpx.Response(
        401, json={"error": {"message": "invalid_api_key"}}, request=httpx.Request("POST", transport.endpoint)
    )
    message = transport._describe_error(response)
    assert "MY_KEY" in message
    assert "not set" in message
    assert "secret" not in message


def test_rejected_key_is_reported_as_rejected_not_missing():
    transport = OpenAiHttpTransport(
        "https://api.example.com/v1", api_key="wrong", api_key_env="MY_KEY"
    )
    response = httpx.Response(
        401, json={"error": {"message": "invalid_api_key"}}, request=httpx.Request("POST", transport.endpoint)
    )
    message = transport._describe_error(response)
    assert "rejected" in message


def test_503_explains_the_queue():
    transport = OpenAiHttpTransport("http://127.0.0.1:8000")
    response = httpx.Response(
        503, json={}, request=httpx.Request("POST", transport.endpoint)
    )
    assert "queue" in transport._describe_error(response)


def test_run_through_prepared_audio_builds_a_multipart_request(monkeypatch, settings, synth_wav):
    """Asserts on the actual outgoing form, which is the hosted-provider contract."""
    captured: dict = {}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> dict:
            return VERBOSE_JSON

    class FakeClient:
        def __init__(self, timeout=None):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, headers=None, data=None, files=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["data"] = data
            captured["files"] = files
            return FakeResponse()

    monkeypatch.setattr(
        "app.transport.openai_http.httpx.Client", FakeClient
    )

    audio = prepare_audio(synth_wav, settings.audio_dir)
    transport = OpenAiHttpTransport(
        "http://127.0.0.1:8000", api_key="k", api_key_env="K"
    )
    result = transport.transcribe(
        StubProvider(),
        audio,
        TranscriptionOptions(word_timestamps=True, keywords=("Ada", "Quillon")),
    )

    assert captured["url"] == "http://127.0.0.1:8000/v1/audio/transcriptions"
    assert captured["headers"] == {"Authorization": "Bearer k"}
    assert captured["data"]["model"] == "phonon-2"
    assert captured["data"]["response_format"] == "verbose_json"
    assert captured["data"]["timestamp_granularities[]"] == "word"
    assert captured["data"]["hotwords"] == "Ada,Quillon"
    # Language selection is not sent when the caller did not ask for it.
    assert "language" not in captured["data"]
    assert captured["files"]["file"][0] == audio.path.name
    assert result.text == VERBOSE_JSON["text"]


def test_timeout_is_reported_as_a_provider_timeout(monkeypatch, settings, synth_wav):
    from app.errors import ProviderTimeout

    class FakeClient:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, *args, **kwargs):
            raise httpx.TimeoutException("too slow")

    monkeypatch.setattr("app.transport.openai_http.httpx.Client", FakeClient)

    audio = prepare_audio(synth_wav, settings.audio_dir)
    transport = OpenAiHttpTransport("http://127.0.0.1:8000", timeout_s=5)
    with pytest.raises(ProviderTimeout):
        transport.transcribe(StubProvider(), audio, TranscriptionOptions())


def test_connection_failure_suggests_starting_the_server(monkeypatch, settings, synth_wav):
    class FakeClient:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, *args, **kwargs):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr("app.transport.openai_http.httpx.Client", FakeClient)

    audio = prepare_audio(synth_wav, settings.audio_dir)
    transport = OpenAiHttpTransport("http://127.0.0.1:8000")
    with pytest.raises(ProviderError) as excinfo:
        transport.transcribe(StubProvider(), audio, TranscriptionOptions())
    assert "Start the local server" in str(excinfo.value)
