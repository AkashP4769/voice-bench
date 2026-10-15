"""Aggregating an evaluation run.

Error rates are **micro-averaged**: total errors divided by total reference units
across every scored clip. Averaging per-clip rates instead would let a five-word
clip count as much as a forty-word one, which is not what a corpus-level word
error rate means, and it is the convention the Open ASR Leaderboard uses.

Clips a provider cannot attempt are counted separately rather than scored as
failures. That distinction is the whole point of including languages one model
does not support: "skipped, English-only" is information, whereas a 100% error
rate would just look like a broken model.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

#: A clip/provider pair that was scored, or why it was not.
OK = "ok"
UNSUPPORTED = "unsupported"
ERROR = "error"

#: Stands in for a value that does not exist, so a missing rate never reads as 0.
DASH = "\u2014"


@dataclass
class ClipResult:
    """One provider's attempt at one clip."""

    clip_id: str
    source: str
    language: str
    provider: str
    status: str = OK
    reference: str = ""
    text: str = ""
    note: str = ""
    wer: float | None = None
    cer: float | None = None
    reference_words: int = 0
    word_errors: int = 0
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    #: Character-level counts, kept separately because CER uses its own units.
    reference_chars: int = 0
    char_errors: int = 0
    latency_ms: float | None = None
    real_time_factor: float | None = None
    audio_duration_ms: float | None = None

    @property
    def scored(self) -> bool:
        return self.status == OK and self.wer is not None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProviderSummary:
    provider: str
    clips_scored: int = 0
    clips_skipped: int = 0
    clips_failed: int = 0
    reference_words: int = 0
    word_errors: int = 0
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    char_errors: int = 0
    reference_chars: int = 0
    #: Micro-averaged, and None rather than 0 when nothing was scored.
    wer: float | None = None
    cer: float | None = None
    mean_latency_ms: float | None = None
    mean_real_time_factor: float | None = None
    audio_s: float = 0.0
    total_latency_ms: float = 0.0
    latencies: int = 0
    rtf_total: float = 0.0
    rtf_count: int = 0
    by_language: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_reasons: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarise(
    provider: str,
    rows: Iterable[ClipResult],
    language_of: dict[str, str] | None = None,
) -> ProviderSummary:
    """Fold one provider's clip results into a summary."""
    summary = ProviderSummary(provider=provider)
    languages: dict[str, dict[str, int]] = {}

    for row in rows:
        if row.status == UNSUPPORTED:
            summary.clips_skipped += 1
            reason = row.note or "unsupported"
            summary.skip_reasons[reason] = summary.skip_reasons.get(reason, 0) + 1
            continue
        if row.status == ERROR or row.wer is None:
            summary.clips_failed += 1
            continue

        summary.clips_scored += 1
        summary.reference_words += row.reference_words
        summary.word_errors += row.word_errors
        summary.substitutions += row.substitutions
        summary.insertions += row.insertions
        summary.deletions += row.deletions
        summary.reference_chars += row.reference_chars
        summary.char_errors += row.char_errors

        if row.latency_ms is not None:
            summary.total_latency_ms += row.latency_ms
            summary.latencies += 1
        if row.real_time_factor is not None:
            summary.rtf_total += row.real_time_factor
            summary.rtf_count += 1
        if row.audio_duration_ms:
            summary.audio_s += row.audio_duration_ms / 1000.0

        bucket = languages.setdefault(
            row.language, {"clips": 0, "reference_words": 0, "word_errors": 0}
        )
        bucket["clips"] += 1
        bucket["reference_words"] += row.reference_words
        bucket["word_errors"] += row.word_errors

    if summary.reference_words:
        summary.wer = summary.word_errors / summary.reference_words
    if summary.reference_chars:
        summary.cer = summary.char_errors / summary.reference_chars
    if summary.latencies:
        summary.mean_latency_ms = summary.total_latency_ms / summary.latencies
    if summary.rtf_count:
        summary.mean_real_time_factor = summary.rtf_total / summary.rtf_count

    for language, bucket in sorted(languages.items()):
        summary.by_language[language] = {
            "clips": bucket["clips"],
            "reference_words": bucket["reference_words"],
            "wer": (
                bucket["word_errors"] / bucket["reference_words"]
                if bucket["reference_words"]
                else None
            ),
        }
    return summary


def summarise_all(
    rows: Iterable[ClipResult], providers: Iterable[str]
) -> list[ProviderSummary]:
    rows = list(rows)
    return [
        summarise(provider, [row for row in rows if row.provider == provider])
        for provider in providers
    ]


def to_json(
    summaries: list[ProviderSummary],
    rows: list[ClipResult],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, float):
            # Keep the JSON readable; the full float adds noise, not precision.
            return round(value, 4)
        return value

    return {
        "metadata": metadata or {},
        "summary": [
            {key: clean(value) for key, value in summary.as_dict().items()}
            for summary in summaries
        ],
        "clips": [
            {key: clean(value) for key, value in row.as_dict().items()}
            for row in rows
        ],
    }


CSV_COLUMNS = [
    "clip_id",
    "source",
    "language",
    "provider",
    "status",
    "note",
    "wer",
    "cer",
    "reference_words",
    "substitutions",
    "insertions",
    "deletions",
    "latency_ms",
    "real_time_factor",
    "audio_duration_ms",
    "reference",
    "text",
]


def to_csv(rows: list[ClipResult]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(CSV_COLUMNS)
    for row in rows:
        payload = row.as_dict()
        writer.writerow([payload.get(column) for column in CSV_COLUMNS])
    return buffer.getvalue()


def render_table(summaries: list[ProviderSummary]) -> str:
    """A fixed-width summary, for reading in a terminal."""
    header = (
        f"{'provider':<12} {'scored':>6} {'skip':>5} {'fail':>5} "
        f"{'words':>7} {'WER':>8} {'CER':>8} {'latency':>10} {'RTF':>7}"
    )
    lines = [header, "-" * len(header)]
    for summary in summaries:
        wer = f"{summary.wer * 100:.2f}%" if summary.wer is not None else DASH
        cer = f"{summary.cer * 100:.2f}%" if summary.cer is not None else DASH
        latency = (
            f"{summary.mean_latency_ms:.0f} ms"
            if summary.mean_latency_ms is not None
            else DASH
        )
        rtf = (
            f"{summary.mean_real_time_factor:.3f}"
            if summary.mean_real_time_factor is not None
            else DASH
        )
        lines.append(
            f"{summary.provider:<12} {summary.clips_scored:>6} "
            f"{summary.clips_skipped:>5} {summary.clips_failed:>5} "
            f"{summary.reference_words:>7} {wer:>8} {cer:>8} "
            f"{latency:>10} {rtf:>7}"
        )
    return "\n".join(lines)


def render_language_table(summaries: list[ProviderSummary]) -> str:
    """Word error rate per language, which is where a capability gap shows up."""
    languages = sorted(
        {
            language
            for summary in summaries
            for language in summary.by_language
        }
    )
    if not languages:
        return ""

    width = max(9, *(len(name) for name in languages))
    header = f"{'language':<{width}}" + "".join(
        f"{summary.provider:>14}" for summary in summaries
    )
    lines = [header, "-" * len(header)]
    for language in languages:
        row = f"{language:<{width}}"
        for summary in summaries:
            bucket = summary.by_language.get(language)
            if bucket and bucket["wer"] is not None:
                row += f"{bucket['wer'] * 100:>13.2f}%"
            else:
                # A language this provider cannot attempt; a dash, not 100%.
                row += f"{DASH:>14}"
        lines.append(row)
    skipped = [
        (summary.provider, summary.skip_reasons)
        for summary in summaries
        if summary.skip_reasons
    ]
    if skipped:
        lines.append("")
        for provider, reasons in skipped:
            detail = ", ".join(f"{count} {reason}" for reason, count in reasons.items())
            lines.append(f"skipped by {provider}: {detail}")
    return "\n".join(lines)
