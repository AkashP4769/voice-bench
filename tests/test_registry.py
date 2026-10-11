"""Provider configuration and the API-key indirection rule."""

from __future__ import annotations

import json

import pytest
import yaml

from app.config import load_settings
from app.providers.registry import UnknownProviderType, capabilities_for, describe_configuration

REMOTE_YAML = """
providers:
  hosted:
    type: phonon
    model: phonon-2
    transport: openai_http
    server:
      base_url: https://api.example.invalid/v1
      api_key_env: HOSTED_SPEECH_KEY
      timeout_s: 120
  local-server:
    type: phonon
    model: phonon-2
    transport: openai_http
    server:
      base_url: http://127.0.0.1:8000
      api_key_env: LOOPBACK_KEY
defaults:
  word_timestamps: true
"""


@pytest.fixture
def remote_settings(tmp_path):
    path = tmp_path / "providers.yaml"
    path.write_text(REMOTE_YAML, encoding="utf-8")
    return load_settings(tmp_path, path)


def test_shipped_configuration_defines_both_models(settings):
    assert {"whistle", "phonon-2"} <= set(settings.providers)


def test_shipped_configuration_starts_local(settings):
    for config in settings.providers.values():
        assert config.transport == "in_process"


def test_remote_provider_is_unconfigured_without_the_key(remote_settings, monkeypatch):
    monkeypatch.delenv("HOSTED_SPEECH_KEY", raising=False)
    assert remote_settings.provider("hosted").credential_configured is False


def test_remote_provider_is_configured_with_the_key(remote_settings, monkeypatch):
    monkeypatch.setenv("HOSTED_SPEECH_KEY", "a-real-key")
    config = remote_settings.provider("hosted")
    assert config.credential_configured is True
    # The key is resolved on demand and is never part of the config object.
    assert config.api_key() == "a-real-key"


def test_loopback_server_needs_no_key(remote_settings, monkeypatch):
    monkeypatch.delenv("LOOPBACK_KEY", raising=False)
    assert remote_settings.provider("local-server").credential_configured is True


def test_description_never_contains_the_secret(remote_settings, monkeypatch):
    """The one rule that must hold no matter how the UI changes."""
    secret = "sk-do-not-leak-this-value"
    monkeypatch.setenv("HOSTED_SPEECH_KEY", secret)
    description = describe_configuration(remote_settings.provider("hosted"))

    assert secret not in json.dumps(description)
    # The variable *name* is reported so the user knows what to set.
    assert description["api_key_env"] == "HOSTED_SPEECH_KEY"
    assert description["credential_configured"] is True


def test_capabilities_available_without_loading_an_engine(remote_settings):
    capabilities = capabilities_for(remote_settings.provider("hosted"))
    assert capabilities.languages == ("en",)
    assert capabilities.word_timestamps is True


def test_unknown_provider_type_is_rejected(remote_settings):
    config = remote_settings.provider("hosted")
    object.__setattr__(config, "type", "does-not-exist")
    with pytest.raises(UnknownProviderType) as excinfo:
        capabilities_for(config)
    assert "unknown provider type" in str(excinfo.value)


def test_unknown_provider_lookup_lists_known_names(settings):
    with pytest.raises(KeyError) as excinfo:
        settings.provider("nope")
    assert "whistle" in str(excinfo.value)


def test_defaults_are_loaded(settings):
    assert settings.defaults.target_sample_rate == 16000
    assert settings.defaults.word_timestamps is True


def test_provider_identity_comes_from_configuration(settings):
    """Results and API requests address a provider by its configured name."""
    from app.providers.registry import build_provider

    provider = build_provider(settings.provider("phonon-2"), settings)
    assert provider.name == "phonon-2"
    assert provider.provider_type == "phonon"


def test_one_adapter_type_can_be_configured_more_than_once(remote_settings):
    """A local and a hosted instance of the same model are distinct providers."""
    from app.providers.registry import build_provider

    hosted = build_provider(remote_settings.provider("hosted"), remote_settings)
    loopback = build_provider(remote_settings.provider("local-server"), remote_settings)

    assert hosted.name == "hosted"
    assert loopback.name == "local-server"
    assert hosted.name != loopback.name
    assert hosted.transport.name == "openai_http"
    assert hosted.transport.endpoint == "https://api.example.invalid/v1/audio/transcriptions"
