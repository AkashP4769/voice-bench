"""Word and character error rates.

The edit-distance implementation counts substitutions, insertions and deletions
rather than only the total, because the breakdown is what tells you how a model
failed: a model that drops words behaves very differently from one that
hallucinates them.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.scoring.normalize import DEFAULT_POLICY, NormalizationPolicy, normalize


@dataclass(frozen=True)
class EditCounts:
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions


@dataclass(frozen=True)
class ScoreResult:
    """Scores for one hypothesis against one reference."""

    reference_units: int
    hypothesis_units: int
    unit_errors: int
    unit_error_rate: float | None
    char_reference_units: int
    char_errors: int
    char_error_rate: float | None
    counts: EditCounts
    policy: str

    def as_dict(self) -> dict:
        return {
            "wer": self.unit_error_rate,
            "cer": self.char_error_rate,
            "reference_words": self.reference_units,
            "hypothesis_words": self.hypothesis_units,
            "substitutions": self.counts.substitutions,
            "insertions": self.counts.insertions,
            "deletions": self.counts.deletions,
            "reference_chars": self.char_reference_units,
            "char_errors": self.char_errors,
            "policy": self.policy,
        }


def edit_counts(reference: list[str], hypothesis: list[str]) -> EditCounts:
    """Levenshtein alignment with an operation breakdown."""
    n, m = len(reference), len(hypothesis)
    if n == 0 and m == 0:
        return EditCounts()

    # dp[i][j] = cheapest way to turn reference[:i] into hypothesis[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ref_unit = reference[i - 1]
        for j in range(1, m + 1):
            if ref_unit == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    substitutions = insertions = deletions = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1
        else:
            deletions += 1
            i -= 1
    return EditCounts(substitutions, insertions, deletions)


def score(
    reference: str,
    hypothesis: str,
    policy: NormalizationPolicy = DEFAULT_POLICY,
) -> ScoreResult:
    """Score ``hypothesis`` against ``reference``.

    An empty reference yields ``None`` rather than zero: a rate against nothing
    is undefined, and reporting 0.0 would read as a perfect score.
    """
    ref_text = normalize(reference, policy)
    hyp_text = normalize(hypothesis, policy)

    ref_words = ref_text.split() if ref_text else []
    hyp_words = hyp_text.split() if hyp_text else []
    word_counts = edit_counts(ref_words, hyp_words)

    raw_reference = normalize(reference, NormalizationPolicy.NONE)
    raw_hypothesis = normalize(hypothesis, NormalizationPolicy.NONE)
    ref_chars = [c for c in raw_reference if not c.isspace()]
    hyp_chars = [c for c in raw_hypothesis if not c.isspace()]
    char_counts = edit_counts(ref_chars, hyp_chars)

    wer = word_counts.errors / len(ref_words) if ref_words else None
    cer = char_counts.errors / len(ref_chars) if ref_chars else None

    return ScoreResult(
        reference_units=len(ref_words),
        hypothesis_units=len(hyp_words),
        unit_errors=word_counts.errors,
        unit_error_rate=wer,
        char_reference_units=len(ref_chars),
        char_errors=char_counts.errors,
        char_error_rate=cer,
        counts=word_counts,
        policy=policy.value,
    )
