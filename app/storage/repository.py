"""Run persistence.

A run is stored as one row plus one row per provider. Provider-native payloads
are kept in a ``raw`` column so a surprising result can be investigated later,
but nothing in the UI reads it.

API key material never reaches the database: the run record stores provider
names, options and outputs, and the environment block stores package versions.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.storage.db import connect, init_schema


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


class Repository:
    """Thread-safe-enough wrapper: SQLite access is serialised by a lock."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._connection = connect(self.db_path)
        init_schema(self._connection)
        self._lock = threading.Lock()

    # -- writes ------------------------------------------------------------

    def save_run(self, run: dict[str, Any]) -> str:
        """Persist a completed run, returning its id.

        The id and timestamp are written back onto the caller's dict so a run
        that was just executed and a run reloaded from storage have the same
        shape and can share one template.
        """
        run_id = run.get("id") or uuid.uuid4().hex[:12]
        created_at = run.get("created_at") or _now()
        run["id"] = run_id
        run["created_at"] = created_at
        audio = run["audio"]
        with self._lock:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO runs (id, created_at, audio_sha256, audio_name,
                                      audio_path, audio_duration_ms,
                                      audio_sample_rate, audio_channels,
                                      providers, options, reference_text,
                                      environment, caveats, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        created_at,
                        audio["sha256"],
                        audio.get("name"),
                        audio["path"],
                        audio.get("duration_ms"),
                        audio.get("sample_rate"),
                        audio.get("channels"),
                        json.dumps(run.get("providers", [])),
                        json.dumps(run.get("options", {})),
                        run.get("reference_text"),
                        json.dumps(run.get("environment", {})),
                        json.dumps(run.get("caveats", [])),
                        run.get("notes"),
                    ),
                )
                for result in run.get("results", []):
                    self._connection.execute(
                        """
                        INSERT INTO results (run_id, provider, model, model_version,
                                             transport, text, language, latency_ms,
                                             time_to_first_token_ms, real_time_factor,
                                             audio_duration_ms, words, windowing,
                                             truncated, error, warnings, raw, score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            result.get("provider"),
                            result.get("model"),
                            result.get("model_version"),
                            result.get("transport"),
                            result.get("text"),
                            result.get("language"),
                            result.get("latency_ms"),
                            result.get("time_to_first_token_ms"),
                            result.get("real_time_factor"),
                            result.get("audio_duration_ms"),
                            json.dumps(result.get("words") or []),
                            json.dumps(result.get("windowing")),
                            None
                            if result.get("truncated") is None
                            else int(bool(result.get("truncated"))),
                            result.get("error"),
                            json.dumps(result.get("warnings") or []),
                            json.dumps(result.get("raw_response") or {}),
                            json.dumps(result.get("score")),
                        ),
                    )
        return run_id

    # -- reads -------------------------------------------------------------

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT r.*, (SELECT COUNT(*) FROM results WHERE run_id = r.id) AS result_count
                FROM runs r ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._summarize(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Load a run in the same shape ``execute_run`` returns."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return None
            result_rows = self._connection.execute(
                "SELECT * FROM results WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "audio": {
                "path": row["audio_path"],
                "name": row["audio_name"],
                "sha256": row["audio_sha256"],
                "duration_ms": row["audio_duration_ms"],
                "sample_rate": row["audio_sample_rate"],
                "channels": row["audio_channels"],
            },
            "providers": _loads(row["providers"], []),
            "options": _loads(row["options"], {}),
            "reference_text": row["reference_text"],
            "environment": _loads(row["environment"], {}),
            "caveats": _loads(row["caveats"], []),
            "notes": row["notes"],
            "results": [self._expand(result) for result in result_rows],
        }

    def delete_run(self, run_id: str) -> bool:
        with self._lock:
            with self._connection:
                cursor = self._connection.execute(
                    "DELETE FROM runs WHERE id = ?", (run_id,)
                )
        return cursor.rowcount > 0

    # -- shaping -----------------------------------------------------------

    def _summarize(self, row: sqlite3.Row) -> dict[str, Any]:
        summary = {
            "id": row["id"],
            "created_at": row["created_at"],
            "audio_name": row["audio_name"],
            "audio_sha256": row["audio_sha256"],
            "audio_duration_ms": row["audio_duration_ms"],
            "providers": _loads(row["providers"], []),
            "options": _loads(row["options"], {}),
            "environment": _loads(row["environment"], {}),
            "notes": row["notes"],
        }
        if "result_count" in row.keys():
            summary["result_count"] = row["result_count"]
        return summary

    def _expand(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "provider": row["provider"],
            "model": row["model"],
            "model_version": row["model_version"],
            "transport": row["transport"],
            "text": row["text"],
            "language": row["language"],
            "latency_ms": row["latency_ms"],
            "time_to_first_token_ms": row["time_to_first_token_ms"],
            "real_time_factor": row["real_time_factor"],
            "audio_duration_ms": row["audio_duration_ms"],
            "words": _loads(row["words"], []),
            "windowing": _loads(row["windowing"], None),
            "truncated": None if row["truncated"] is None else bool(row["truncated"]),
            "error": row["error"],
            "warnings": _loads(row["warnings"], []),
            "raw_response": _loads(row["raw"], {}),
            "score": _loads(row["score"], None),
        }
