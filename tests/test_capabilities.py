"""Declared capabilities, which are what the UI and orchestrator trust."""

from __future__ import annotations

from app.domain.models import TranscriptionOptions
from app.providers import phonon, whistle


def test_whistle_declares_its_seven_languages():
    assert whistle.CAPABILITIES.languages == ("en", "de", "fr", "es", "it", "nl", "pl")
    assert whistle.CAPABILITIES.supports_language_selection is True


def test_whistle_does_not_declare_a_duration_limit():
    """Verified: the engine transcribed a 47 s clip complete in one call."""
    assert whistle.CAPABILITIES.max_duration_s is None


def test_whistle_accepts_a_supported_language():
    assert whistle.CAPABILITIES.accepts(TranscriptionOptions(language="de")) == []


def test_whistle_rejects_an_unsupported_language():
    problems = whistle.CAPABILITIES.accepts(TranscriptionOptions(language="zz"))
    assert any("not supported" in problem for problem in problems)


def test_phonon_is_english_only_and_cannot_select_language():
    assert phonon.CAPABILITIES.languages == ("en",)
    assert phonon.CAPABILITIES.supports_language_selection is False
    problems = phonon.CAPABILITIES.accepts(TranscriptionOptions(language="de"))
    assert any("not supported" in problem for problem in problems)


def test_phonon_enforces_the_twenty_five_hotword_ceiling():
    too_many = TranscriptionOptions(keywords=tuple(f"term{i}" for i in range(26)))
    problems = phonon.CAPABILITIES.accepts(too_many)
    assert any("25" in problem for problem in problems)

    within = TranscriptionOptions(keywords=tuple(f"term{i}" for i in range(25)))
    assert phonon.CAPABILITIES.accepts(within) == []


def test_both_providers_offer_word_timestamps_and_hotwords():
    for capabilities in (whistle.CAPABILITIES, phonon.CAPABILITIES):
        assert capabilities.word_timestamps is True
        assert capabilities.hotwords is True


def test_phonon_does_not_claim_streaming_it_does_not_expose():
    assert phonon.CAPABILITIES.streaming is False


def test_capabilities_serialize():
    payload = whistle.CAPABILITIES.as_dict()
    assert payload["languages"] == ["en", "de", "fr", "es", "it", "nl", "pl"]
    assert payload["max_duration_s"] is None
