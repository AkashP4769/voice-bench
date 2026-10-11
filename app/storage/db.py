"""SQLite schema, migrations and connection handling.

SQLite is enough for a single-machine benchmark harness, and a repository layer
keeps a later move to Postgres from touching the API or the services.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

#: Bumped whenever the shape below changes, so migrations run once.
SCHEMA_VERSION = 2

RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    id                TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    -- Null for a live session: the transcript text is kept, the audio is not. An
    -- empty string would be a sentinel pretending to be a hash, so these are
    -- genuinely nullable and the UI states that the audio was not retained.
    audio_sha256      TEXT,
    audio_name        TEXT,
    audio_path        TEXT,
    audio_duration_ms REAL,
    audio_sample_rate INTEGER,
    audio_channels    INTEGER,
    source            TEXT NOT NULL DEFAULT 'file',
    providers         TEXT NOT NULL,
    options           TEXT NOT NULL,
    reference_text    TEXT,
    environment       TEXT NOT NULL,
    caveats           TEXT,
    notes             TEXT
);
"""

RESULTS_TABLE = """
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
    score                 TEXT,
    stream_metrics        TEXT
);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS idx_results_run_id ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC);
"""

SCHEMA = RUNS_TABLE + RESULTS_TABLE + INDEXES

# Columns added after the first release. A harness that keeps its history should
# keep working when the schema grows, so these are added in place.
_ADDITIONS = {
    "results": {
        "stream_metrics": "TEXT",
    },
}

# Columns carried across the v1 -> v2 rebuild of `runs`. `source` is absent here
# on purpose: rows migrated from v1 are file runs, which is the column default.
_RUNS_COLUMNS = (
    "id, created_at, audio_sha256, audio_name, audio_path, audio_duration_ms,"
    " audio_sample_rate, audio_channels, providers, options, reference_text,"
    " environment, caveats, notes"
)


def connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA)
    _migrate(connection)
    _add_missing_columns(connection)
    connection.commit()


def _columns(connection: sqlite3.Connection, table: str) -> dict[str, sqlite3.Row]:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return {row["name"]: row for row in rows}


def _needs_rebuild(connection: sqlite3.Connection) -> bool:
    columns = _columns(connection, "runs")
    if not columns:
        # No table yet; the schema just created the current shape.
        return False
    if "source" not in columns:
        return True
    audio_hash = columns.get("audio_sha256")
    return bool(audio_hash is not None and audio_hash["notnull"])


def _migrate(connection: sqlite3.Connection) -> None:
    """Bring an existing database up to the current schema.

    v1 stored ``audio_sha256`` and ``audio_path`` as NOT NULL, which cannot
    express a live session that has no audio. SQLite cannot drop NOT NULL in
    place, so the table is rebuilt with foreign keys off, which is the documented
    approach for a schema change of this kind.
    """
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version >= SCHEMA_VERSION and not _needs_rebuild(connection):
        return
    if not _needs_rebuild(connection):
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        return

    connection.execute("PRAGMA foreign_keys = OFF")
    try:
        with connection:
            connection.execute("DROP TABLE IF EXISTS runs_migrating")
            connection.execute(
                RUNS_TABLE.replace(
                    "CREATE TABLE IF NOT EXISTS runs", "CREATE TABLE runs_migrating"
                )
            )
            connection.execute(
                f"INSERT INTO runs_migrating ({_RUNS_COLUMNS})"
                f" SELECT {_RUNS_COLUMNS} FROM runs"
            )
            connection.execute("DROP TABLE runs")
            connection.execute("ALTER TABLE runs_migrating RENAME TO runs")
            connection.executescript(INDEXES)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    finally:
        connection.execute("PRAGMA foreign_keys = ON")


def _add_missing_columns(connection: sqlite3.Connection) -> None:
    for table, columns in _ADDITIONS.items():
        existing = set(_columns(connection, table))
        for name, declaration in columns.items():
            if name not in existing:
                connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN {name} {declaration}"
                )
