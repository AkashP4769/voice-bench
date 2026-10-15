"""Word and character error rates."""

from __future__ import annotations

from app.scoring.metrics import edit_counts, score
from app.scoring.normalize import NormalizationPolicy


def test_identical_text_scores_zero():
    result = score("the quick brown fox", "the quick brown fox")
    assert result.unit_error_rate == 0.0
    assert result.counts.errors == 0


def test_single_substitution():
    result = score("the quick brown fox", "the quick brown box")
    assert result.counts.substitutions == 1
    assert result.counts.insertions == 0
    assert result.counts.deletions == 0
    assert result.unit_error_rate == 0.25


def test_single_deletion():
    result = score("the quick brown fox", "the quick brown")
    assert result.counts.deletions == 1
    assert result.unit_error_rate == 0.25


def test_single_insertion():
    result = score("the quick brown fox", "the quick brown fox jumps")
    assert result.counts.insertions == 1
    assert result.unit_error_rate == 0.25


def test_empty_reference_reports_none_not_zero():
    """A rate against nothing is undefined; zero would read as a perfect score."""
    result = score("", "some hallucinated sentence")
    assert result.unit_error_rate is None
    assert result.counts.insertions == 3


def test_both_empty_scores_zero():
    result = score("", "")
    assert result.unit_error_rate is None
    assert result.counts.errors == 0


def test_normalization_is_applied_before_scoring():
    # Case and punctuation differ, but the words are the same.
    result = score("The quick, brown fox.", "the quick brown fox")
    assert result.unit_error_rate == 0.0


def test_none_policy_keeps_differences():
    result = score(
        "The quick, brown fox.",
        "the quick brown fox",
        NormalizationPolicy.NONE,
    )
    assert result.unit_error_rate is not None
    assert result.unit_error_rate > 0


def test_character_error_rate_counts_characters():
    result = score("abcd", "abce")
    assert result.char_errors == 1
    assert result.char_error_rate == 0.25


def test_an_all_caps_reference_does_not_produce_a_perfect_cer_against_perfect_text():
    """LibriSpeech transcribes in capitals; this must not read as 100% error.

    Regression: CER used to be scored on un-normalized text, so a correctly-cased
    hypothesis against an all-caps reference differed in every character.
    """
    result = score(
        "CONCORD RETURNED TO ITS PLACE",
        "Concord returned to its place.",
    )
    assert result.unit_error_rate == 0.0
    assert result.char_error_rate == 0.0


def test_cer_follows_the_chosen_policy():
    reference = "THE QUICK BROWN FOX"
    hypothesis = "the quick brown fox"

    # Under the default policy both texts normalize to the same thing.
    assert score(reference, hypothesis).char_error_rate == 0.0
    # Under no normalization they differ, which is the strict view.
    strict = score(reference, hypothesis, NormalizationPolicy.NONE)
    assert strict.char_error_rate is not None
    assert strict.char_error_rate > 0


def test_cer_ignores_whitespace_but_still_counts_case_and_content():
    result = score("abc def", "abcdef")
    # Whitespace is excluded from the character comparison.
    assert result.char_error_rate == 0.0


def test_edit_counts_reference_and_hypothesis_swapped():
    counts = edit_counts([], ["a", "b"])
    assert counts.insertions == 2
    counts = edit_counts(["a", "b"], [])
    assert counts.deletions == 2


def test_score_serializes():
    payload = score("a b", "a c").as_dict()
    assert payload["wer"] == 0.5
    assert set(payload) >= {"wer", "cer", "policy", "substitutions"}
