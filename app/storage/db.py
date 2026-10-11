"""SQLite schema and connection handling.

SQLite is enough for a single-machine benchmark harness, and a repository layer
keeps a later move to Postgres from touching the API or the services.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    audio_sha256      TEXT NOT NULL,
    audio_name        TEXT,
    audio_path        TEXT NOT NULL,
    audio_duration_ms REAL,
    audio_sample_rate INTEGER,
    audio_channels    INTEGER,
    providers         TEXT NOT NULL,
    options           TEXT NOT NULL,
    reference_text    TEXT,
    environment       TEXT NOT NULL,
    caveats           TEXT,
    notes             TEXT
);

CREATE TABLE IF NOT EXISTS results (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    provider              TEXT NOT NULL,
    model                 TEXT,
    model_version         TEXT,
    transport             TEXT,
    text                  TEXT,
    language              TEXT,
    latency_ms            REAL,
    time_to_first_token_ms REAL,
    real_time_factor      REAL,
    audio_duration_ms     REAL,
    words                 TEXT,
    windowing             TEXT,
    truncated             INTEGER,
    error                 TEXT,
    warnings              TEXT,
    raw                   TEXT,
    score                 TEXT
);

CREATE INDEX IF NOT EXISTS idx_results_run_id ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA)
    _add_missing_columns(connection)
    connection.commit()


# Columns added after the first release. A harness that keeps its history
# should keep working when the schema grows, so these are added in place.
_ADDITIONS = {
    "runs": {
        "audio_sample_rate": "INTEGER",
        "audio_channels": "INTEGER",
        "caveats": "TEXT",
    },
}


def _add_missing_columns(connection: sqlite3.Connection) -> None:
    for table, columns in _ADDITIONS.items():
        existing = {
            row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        for name, declaration in columns.items():
            if name not in existing:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
