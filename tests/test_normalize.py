"""Scoring normalization."""

from __future__ import annotations

from app.scoring.normalize import (
    NormalizationPolicy,
    language_code,
    normalize,
    strip_punctuation,
)


def test_language_name_maps_to_code():
    # Phonon's server reports "english"; Whistle reports "en".
    assert language_code("english") == "en"
    assert language_code("German") == "de"
    assert language_code("en") == "en"
    assert language_code("pl") == "pl"
    assert language_code(None) is None
    assert language_code("") is None


def test_language_unknown_name_is_passed_through_lowercased():
    assert language_code("Klingon") == "klingon"


def test_strip_punctuation_keeps_apostrophes_and_hyphens():
    assert strip_punctuation("don't") == "don't"
    assert strip_punctuation("well-known") == "well-known"
    assert " " in strip_punctuation("hello, world!")


def test_normalize_policies():
    text = "  The Quick, Brown Fox!  "
    assert normalize(text, NormalizationPolicy.NONE) == "The Quick, Brown Fox!"
    assert normalize(text, NormalizationPolicy.LOWER) == "the quick, brown fox!"
    assert (
        normalize(text, NormalizationPolicy.LOWER_STRIP_PUNCTUATION)
        == "the quick brown fox"
    )


def test_normalize_collapses_whitespace():
    assert normalize("a   b \n c", NormalizationPolicy.LOWER) == "a b c"


def test_normalize_empty():
    assert normalize("") == ""
    assert normalize(None or "") == ""
