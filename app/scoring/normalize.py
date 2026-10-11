"""Text normalization for scoring.

Providers differ in punctuation, casing and numerals, so comparing raw output
would measure formatting rather than accuracy. Everything is normalized before
scoring, and the chosen policy is recorded on the run.

This is the harness's own normalizer, deliberately not a reimplementation of
Whisper's ``EnglishTextNormalizer``. Whisper's version expands numerals and
spelled-out numbers and applies a long list of substitutions; claiming parity
without porting it would be misleading, so the policy is named plainly and can
be swapped once a reference implementation is available.
"""

from __future__ import annotations

import re
import unicodedata
from enum import Enum


class NormalizationPolicy(str, Enum):
    """How text is prepared before scoring."""

    NONE = "none"
    LOWER = "lower"
    LOWER_STRIP_PUNCTUATION = "lower_strip_punctuation"


# One definition of the default, so the CLI, the API and the UI cannot drift
# apart on what "no normalization specified" means.
DEFAULT_POLICY = NormalizationPolicy.LOWER_STRIP_PUNCTUATION


LANGUAGE_NAMES: dict[str, str] = {
    "english": "en",
    "german": "de",
    "french": "fr",
    "spanish": "es",
    "italian": "it",
    "dutch": "nl",
    "polish": "pl",
    "portuguese": "pt",
    "russian": "ru",
    "chinese": "zh",
    "japanese": "ja",
    "korean": "ko",
    "arabic": "ar",
    "hindi": "hi",
    "turkish": "tr",
    "ukrainian": "uk",
    "swedish": "sv",
    "danish": "da",
    "norwegian": "no",
    "finnish": "fi",
    "czech": "cs",
    "greek": "el",
    "hebrew": "he",
    "hungarian": "hu",
    "romanian": "ro",
    "vietnamese": "vi",
    "thai": "th",
    "indonesian": "id",
    "malay": "ms",
    "catalan": "ca",
    "croatian": "hr",
    "slovak": "sk",
    "slovenian": "sl",
    "bulgarian": "bg",
    "estonian": "et",
    "latvian": "lv",
    "lithuanian": "lt",
    "serbian": "sr",
}


def language_code(value: str | None) -> str | None:
    """Reduce a language name or code to a short lowercase code.

    OpenAI-compatible servers report ``"english"`` where Whistle reports
    ``"en"``, so both have to land on the same value before comparison.
    """
    if not value:
        return None
    text = value.strip().lower()
    if not text:
        return None
    if len(text) <= 3 and text.isalpha():
        return text
    return LANGUAGE_NAMES.get(text, text)


def strip_punctuation(text: str) -> str:
    """Drop punctuation, keeping characters that carry meaning.

    Apostrophes and hyphens are kept because removing them merges words that
    the reference transcript also spells with them.
    """
    kept = []
    for char in text:
        category = unicodedata.category(char)
        if char in "'-":
            kept.append(char)
        elif category.startswith("P") or category.startswith("S"):
            kept.append(" ")
        else:
            kept.append(char)
    return "".join(kept)


def normalize(
    text: str,
    policy: NormalizationPolicy = DEFAULT_POLICY,
) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    if policy is NormalizationPolicy.NONE:
        return normalized.strip()
    normalized = normalized.lower()
    if policy is NormalizationPolicy.LOWER:
        return re.sub(r"\s+", " ", normalized).strip()
    normalized = strip_punctuation(normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize(text)
    return normalized.split() if normalized else []
