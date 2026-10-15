"""Persisting automated benchmark runs.

An automated run that exists only in a rendered page cannot be compared with the
run from last week, so every run is written to disk: an immutable copy under
``runs/`` plus a ``latest`` pair holding the newest numbers. The payload is the
same JSON the CLI writes with ``bench evaluate --json-out``, so the page and the
CLI describe a result in exactly one way.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from app.evaluation.report import ClipResult, ProviderSummary, to_csv, to_json

#: Bumped when the stored payload shape changes, so an older file is recognised
#: rather than silently misread.
SCHEMA_VERSION = 1

RUNS_DIRNAME = "runs"
LATEST_STEM = "latest"

#: Run ids become file names, so they are restricted to a safe alphabet.
RUN_ID_PATTERN = re.compile(r"^[0-9A-Za-z][0-9A-Za-z_-]{0,63}$")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build_benchmark_payload(
    *,
    benchmark_key: str,
    benchmark_name: str,
    source_keys: Iterable[str],
    providers: Iterable[str],
    normalization: str,
    summaries: list[ProviderSummary],
    rows: list[ClipResult],
    clips: list[Any],
    manifest: Path | str,
    fetched_clips: int,
    fetch_failures: Iterable[str] = (),
    warm_failures: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble the stored payload for one automated benchmark run."""
    metadata = {
        "schema": SCHEMA_VERSION,
        "created_at": _now(),
        "benchmark": benchmark_key,
        "benchmark_name": benchmark_name,
        "sources": list(source_keys),
        "providers": list(providers),
        "normalization": normalization,
        "clips": len(clips),
        "fetched_clips": fetched_clips,
        "manifest": str(manifest),
        "fetch_failures": list(fetch_failures),
        "warm_failures": dict(warm_failures or {}),
    }
    return to_json(summaries, rows, metadata=metadata)


@dataclass(frozen=True)
class StoredBenchmarkRun:
    """A saved run as it exists on disk."""

    run_id: str
    created_at: str
    benchmark: str
    benchmark_name: str
    providers: list[str]
    normalization: str
    sources: list[str]
    clips: int
    fetched_clips: int
    summaries: list[dict[str, Any]] = field(default_factory=list)
    json_path: str = ""
    csv_path: str | None = None

    def wer_of(self, provider: str) -> float | None:
        for summary in self.summaries:
            if summary.get("provider") == provider:
                return summary.get("wer")
        return None


def _runs_dir(out_dir: Path) -> Path:
    return Path(out_dir) / RUNS_DIRNAME


def _unique_run_id(runs_dir: Path, created_at: str) -> str:
    """A sortable id, with a suffix when two runs share a second."""
    try:
        stamp = datetime.fromisoformat(created_at).strftime("%Y%m%d-%H%M%S")
    except ValueError:
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    candidate = stamp
    suffix = 1
    while (runs_dir / f"{candidate}.json").exists():
        candidate = f"{stamp}-{suffix}"
        suffix += 1
    return candidate


def _rows_to_csv(payload: dict[str, Any]) -> str:
    """Reuse the one CSV definition rather than duplicating the columns.

    The stored clips are already plain dicts, so they are rebuilt into the same
    record type the CLI exports. Floats are the rounded values the JSON holds.
    """
    rows = [ClipResult(**clip) for clip in payload.get("clips") or []]
    return to_csv(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_benchmark_run(out_dir: Path, payload: dict[str, Any]) -> StoredBenchmarkRun:
    """Write a run to ``runs/`` and refresh the ``latest`` pair."""
    out_dir = Path(out_dir)
    runs_dir = _runs_dir(out_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    metadata = payload.setdefault("metadata", {})
    metadata.setdefault("schema", SCHEMA_VERSION)
    created_at = metadata.get("created_at") or _now()
    metadata["created_at"] = created_at
    run_id = metadata.get("run_id") or _unique_run_id(runs_dir, created_at)
    metadata["run_id"] = run_id

    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    csv_text = _rows_to_csv(payload)

    _write_text(runs_dir / f"{run_id}.json", json_text)
    _write_text(runs_dir / f"{run_id}.csv", csv_text)
    # The newest numbers stay easy to find without knowing the run id.
    _write_text(out_dir / f"{LATEST_STEM}.json", json_text)
    _write_text(out_dir / f"{LATEST_STEM}.csv", csv_text)

    return describe_benchmark_run(payload, runs_dir / f"{run_id}.json")


def describe_benchmark_run(payload: dict[str, Any], json_path: Path) -> StoredBenchmarkRun:
    metadata = payload.get("metadata") or {}
    csv_path = json_path.with_suffix(".csv")
    return StoredBenchmarkRun(
        run_id=str(metadata.get("run_id") or json_path.stem),
        created_at=str(metadata.get("created_at") or ""),
        benchmark=str(metadata.get("benchmark") or ""),
        benchmark_name=str(metadata.get("benchmark_name") or metadata.get("benchmark") or ""),
        providers=list(metadata.get("providers") or []),
        normalization=str(metadata.get("normalization") or ""),
        sources=list(metadata.get("sources") or []),
        clips=int(metadata.get("clips") or 0),
        fetched_clips=int(metadata.get("fetched_clips") or 0),
        summaries=[
            {
                "provider": summary.get("provider"),
                "wer": summary.get("wer"),
                "cer": summary.get("cer"),
                "clips_scored": summary.get("clips_scored"),
                "clips_skipped": summary.get("clips_skipped"),
                "clips_failed": summary.get("clips_failed"),
            }
            for summary in payload.get("summary") or []
        ],
        json_path=str(json_path),
        csv_path=str(csv_path) if csv_path.exists() else None,
    )


def list_benchmark_runs(out_dir: Path, limit: int = 10) -> list[StoredBenchmarkRun]:
    """Newest first. A corrupt file is skipped rather than breaking the list."""
    runs_dir = _runs_dir(out_dir)
    if not runs_dir.exists():
        return []

    stored: list[StoredBenchmarkRun] = []
    for path in sorted(runs_dir.glob("*.json"), key=lambda item: item.stem, reverse=True):
        if len(stored) >= max(0, limit):
            break
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        stored.append(describe_benchmark_run(payload, path))
    return stored


def load_benchmark_run(out_dir: Path, run_id: str) -> dict[str, Any] | None:
    """Read one stored run.

    The id is validated before it reaches the filesystem: it arrives from a URL
    and becomes a path segment, so ``..`` and separators must not be accepted.
    """
    if not RUN_ID_PATTERN.match(run_id or ""):
        return None
    path = _runs_dir(out_dir) / f"{run_id}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
