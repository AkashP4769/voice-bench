"""The realtime WebSocket, the save endpoint, and the storage migration.

The engine is stubbed here: these tests are about the plumbing — fan-out,
event relay, refusal paths, and storing a session that has no audio.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from app.streaming.events import StreamEvent, StreamMetrics, StreamResult


class FakeSession:
    """Stands in for a real streaming session."""

    implementation = "native"

    def __init__(self, *, text: str = "hello there") -> None:
        self.text = text
        self.feeds = 0
        self.closed = False

    def feed(self, samples):
        self.feeds += 1
        return [
            StreamEvent.partial(
                "hello",
                pending="there",
                received_s=samples.size / 16000,
                pass_ms=12.5,
            )
        ]

    def stop(self):
        self.closed = True
        return StreamResult(
            provider="",
            text=self.text,
            metrics=StreamMetrics(chunks=self.feeds, audio_s=0.25, decodes=1,
                                  mean_pass_ms=12.5),
            implementation=self.implementation,
        )


class FakePool:
    def __init__(self, sessions=None, error=None) -> None:
        self.sessions = sessions or {}
        self.error = error
        self.opened: list[str] = []

    def open_stream(self, name, options):
        self.opened.append(name)
        if self.error:
            raise self.error
        return self.sessions.setdefault(name, FakeSession())


def _wire(monkeypatch, settings, pool):
    from app.api import routes_stream

    monkeypatch.setattr(routes_stream, "get_settings_from_socket", lambda: settings)
    monkeypatch.setattr(routes_stream, "get_worker_pool_direct", lambda s: pool)


def _drain(ws, limit: int = 80) -> list[dict]:
    """Read until the session ends.

    The caller must have sent a stop frame: the server only emits `ended` then,
    and an unbounded read would otherwise wait forever.
    """
    messages = []
    for _ in range(limit):
        message = ws.receive_json()
        messages.append(message)
        if message.get("type") == "ended":
            break
    return messages


frame = np.zeros(4000, dtype="<f4").tobytes()


# -- happy path ------------------------------------------------------------


def test_session_streams_partials_and_ends_with_transcripts(monkeypatch, settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    pool = FakePool()
    _wire(monkeypatch, settings, pool)
    app = create_app()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/realtime") as ws:
            ws.send_json(
                {"type": "start", "providers": ["whistle"], "sampleRate": 16000}
            )
            ready = ws.receive_json()
            assert ready["type"] == "ready"
            assert ready["providers"] == ["whistle"]
            assert ready["sampleRate"] == 16000
            assert ready["implementations"] == {"whistle": "native"}

            ws.send_bytes(frame)
            ws.send_json({"type": "stop"})
            messages = _drain(ws)

    kinds = [message["type"] for message in messages]
    assert "partial" in kinds
    assert kinds[-1] == "ended"

    partial = next(m for m in messages if m["type"] == "partial")
    assert partial["provider"] == "whistle"
    assert partial["text"] == "hello"
    assert partial["pending"] == "there"
    assert partial["passMs"] == 12.5

    status = next(m for m in messages if m["type"] == "status")
    assert status["state"] == "streaming"

    ended = messages[-1]
    assert ended["transcripts"] == {"whistle": "hello there"}
    assert ended["implementations"] == {"whistle": "native"}
    assert ended["metrics"]["whistle"]["chunks"] >= 1


def test_two_models_run_concurrently(monkeypatch, settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    pool = FakePool()
    _wire(monkeypatch, settings, pool)
    app = create_app()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/realtime") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "providers": ["whistle", "phonon-2"],
                    "sampleRate": 16000,
                }
            )
            assert ws.receive_json()["type"] == "ready"
            ws.send_bytes(frame)
            ws.send_json({"type": "stop"})
            messages = _drain(ws)

    # Both providers were opened, and both appear in the final payload.
    assert set(pool.opened) == {"whistle", "phonon-2"}
    ended = messages[-1]
    assert set(ended["transcripts"]) == {"whistle", "phonon-2"}

    # And both produced their own partials, tagged correctly.
    senders = {
        message["provider"] for message in messages if message["type"] == "partial"
    }
    assert senders == {"whistle", "phonon-2"}


def test_word_timestamp_request_is_flagged_for_phonon(monkeypatch, settings):
    """Phonon's live session has no word timings; say so instead of pretending."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    _wire(monkeypatch, settings, FakePool())
    app = create_app()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/realtime") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "providers": ["phonon-2"],
                    "wordTimestamps": True,
                    "sampleRate": 16000,
                }
            )
            ws.receive_json()
            ws.send_json({"type": "stop"})
            messages = _drain(ws, limit=20)

    statuses = [m for m in messages if m["type"] == "status"]
    assert any(m.get("detail") and "word timestamps" in m["detail"] for m in statuses)


# -- refusals --------------------------------------------------------------


def test_first_message_must_be_a_start_frame(monkeypatch, settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    _wire(monkeypatch, settings, FakePool())
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws/realtime") as ws:
            ws.send_json({"type": "nonsense"})
            message = ws.receive_json()
    assert message["type"] == "error"
    assert "start frame" in message["message"]


def test_selecting_no_provider_is_refused(monkeypatch, settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    _wire(monkeypatch, settings, FakePool())
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws/realtime") as ws:
            ws.send_json({"type": "start", "providers": []})
            message = ws.receive_json()
    assert "at least one provider" in message["message"]


def test_unknown_provider_is_refused_with_the_known_names(monkeypatch, settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    _wire(monkeypatch, settings, FakePool())
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws/realtime") as ws:
            ws.send_json({"type": "start", "providers": ["ghost"]})
            message = ws.receive_json()
    assert "ghost" in message["message"]
    assert "whistle" in message["message"]


def test_wrong_sample_rate_is_refused_rather_than_guessed(monkeypatch, settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    _wire(monkeypatch, settings, FakePool())
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws/realtime") as ws:
            ws.send_json(
                {"type": "start", "providers": ["whistle"], "sampleRate": 48000}
            )
            message = ws.receive_json()
    assert "16000" in message["message"]


def test_a_provider_that_cannot_open_a_session_reports_the_reason(
    monkeypatch, settings
):
    """One provider failing must not take the session down with it."""
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.errors import ProviderError

    pool = FakePool(error=ProviderError("whistle is busy with another request"))
    _wire(monkeypatch, settings, pool)

    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws/realtime") as ws:
            ws.send_json(
                {"type": "start", "providers": ["whistle"], "sampleRate": 16000}
            )
            ready = ws.receive_json()

    assert ready["type"] == "ready"
    assert "whistle" in ready["failed"]
    assert "busy" in ready["failed"]["whistle"]


# -- saving a session ------------------------------------------------------


def test_saving_a_session_stores_text_without_audio(client, repository):
    response = client.post(
        "/api/realtime/save",
        json={
            "providers": ["whistle", "phonon-2"],
            "transcripts": {"whistle": "the quick brown fox", "phonon-2": "the quick brown fox"},
            "metrics": {"whistle": {"mean_pass_ms": 1200.0, "audio_s": 6.0}},
            "durationMs": 6000,
            "wordTimestamps": True,
            "cadenceSeconds": 2.0,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["saved"] is True

    run = client.get(f"/api/runs/{payload['runId']}").json()
    assert run["source"] == "realtime"
    # No audio was kept, and the record says so rather than faking an identity.
    assert run["audio"] is None
    assert run["duration_ms"] == 6000
    assert {r["provider"] for r in run["results"]} == {"whistle", "phonon-2"}
    assert run["results"][0]["transport"] == "stream"
    assert any("not retained" in caveat for caveat in run["caveats"])


def test_saved_session_carries_stream_metrics(client):
    response = client.post(
        "/api/realtime/save",
        json={
            "providers": ["whistle"],
            "transcripts": {"whistle": "hello"},
            "metrics": {"whistle": {"mean_pass_ms": 900.0, "audio_s": 12.0,
                                    "realtime_factor": 0.63, "segments": 4}},
            "durationMs": 12000,
        },
    )
    run_id = response.json()["runId"]
    run = client.get(f"/api/runs/{run_id}").json()
    metrics = run["results"][0]["metrics"]
    assert metrics["realtime_factor"] == 0.63
    assert metrics["segments"] == 4


def test_saved_session_is_scored_when_a_reference_is_given(client):
    response = client.post(
        "/api/realtime/save",
        json={
            "providers": ["whistle"],
            "transcripts": {"whistle": "the quick brown box"},
            "referenceText": "the quick brown fox",
            "durationMs": 3000,
        },
    )
    run = client.get(f"/api/runs/{response.json()['runId']}").json()
    score = run["results"][0]["score"]
    assert score is not None
    assert score["wer"] == 0.25


def test_realtime_run_exports_without_audio(client):
    response = client.post(
        "/api/realtime/save",
        json={
            "providers": ["whistle"],
            "transcripts": {"whistle": "hello there"},
            "durationMs": 2000,
        },
    )
    run_id = response.json()["runId"]

    as_json = client.get(f"/api/runs/{run_id}/export?format=json")
    assert as_json.status_code == 200
    assert as_json.json()["audio"] is None

    as_csv = client.get(f"/api/runs/{run_id}/export?format=csv")
    assert as_csv.status_code == 200
    assert "hello there" in as_csv.text


def test_realtime_run_page_renders_without_audio(client):
    response = client.post(
        "/api/realtime/save",
        json={
            "providers": ["whistle"],
            "transcripts": {"whistle": "spoken words"},
            "durationMs": 4000,
        },
    )
    page = client.get(f"/runs/{response.json()['runId']}")
    assert page.status_code == 200
    assert "spoken words" in page.text
    assert "audio not retained" in page.text


def test_history_marks_a_realtime_run(client):
    client.post(
        "/api/realtime/save",
        json={"providers": ["whistle"], "transcripts": {"whistle": "x"},
              "durationMs": 1000},
    )
    history = client.get("/history").text
    assert "realtime" in history


def test_saving_without_providers_is_refused(client):
    payload = client.post("/api/realtime/save", json={"providers": []}).json()
    assert payload["saved"] is False


# -- migration -------------------------------------------------------------


V1_SCHEMA = """
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    audio_sha256 TEXT NOT NULL,
    audio_name TEXT,
    audio_path TEXT NOT NULL,
    audio_duration_ms REAL,
    providers TEXT NOT NULL,
    options TEXT NOT NULL,
    reference_text TEXT,
    environment TEXT NOT NULL,
    notes TEXT
);
CREATE TABLE results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model TEXT, model_version TEXT, transport TEXT, text TEXT, language TEXT,
    latency_ms REAL, time_to_first_token_ms REAL, real_time_factor REAL,
    audio_duration_ms REAL, words TEXT, windowing TEXT, truncated INTEGER,
    error TEXT, warnings TEXT, raw TEXT, score TEXT
);
"""


def _make_v1_database(path):
    connection = sqlite3.connect(str(path))
    connection.executescript(V1_SCHEMA)
    connection.execute(
        """
        INSERT INTO runs (id, created_at, audio_sha256, audio_name, audio_path,
                          audio_duration_ms, providers, options, reference_text,
                          environment, notes)
        VALUES ('old1', '2026-01-01T00:00:00+00:00', 'abc123', 'clip.wav',
                'data/clip.wav', 1234.0, '["whistle"]', '{}', NULL, '{}', 'legacy')
        """
    )
    connection.execute(
        "INSERT INTO results (run_id, provider, text) VALUES ('old1', 'whistle', 'old text')"
    )
    connection.commit()
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()


def test_migration_makes_audio_nullable_and_keeps_existing_rows(tmp_path):
    from app.storage.db import SCHEMA_VERSION, connect, init_schema

    db_path = tmp_path / "legacy.sqlite3"
    _make_v1_database(db_path)

    connection = connect(db_path)
    init_schema(connection)

    # The old run survived, and is marked as a file run by the column default.
    row = connection.execute("SELECT * FROM runs WHERE id = 'old1'").fetchone()
    assert row is not None
    assert row["audio_sha256"] == "abc123"
    assert row["source"] == "file"
    # Its results came through too, and gained the new column.
    result = connection.execute(
        "SELECT * FROM results WHERE run_id = 'old1'"
    ).fetchone()
    assert result["text"] == "old text"
    assert result["stream_metrics"] is None
    assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION

    # And an audio-less run is now possible.
    connection.execute(
        """
        INSERT INTO runs (id, created_at, audio_sha256, audio_path, source,
                          providers, options, environment)
        VALUES ('new1', '2026-02-01T00:00:00+00:00', NULL, NULL, 'realtime',
                '["whistle"]', '{}', '{}')
        """
    )
    connection.commit()
    assert connection.execute(
        "SELECT audio_sha256 FROM runs WHERE id = 'new1'"
    ).fetchone()["audio_sha256"] is None
    connection.close()


def test_migration_is_idempotent(tmp_path):
    from app.storage.db import connect, init_schema

    db_path = tmp_path / "legacy.sqlite3"
    _make_v1_database(db_path)

    connection = connect(db_path)
    init_schema(connection)
    init_schema(connection)  # must not raise or duplicate anything

    assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    connection.close()


def test_fresh_database_needs_no_migration(tmp_path):
    from app.storage.db import connect, init_schema

    connection = connect(tmp_path / "fresh.sqlite3")
    init_schema(connection)
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(runs)")}
    assert "source" in columns
    audio = next(
        row for row in connection.execute("PRAGMA table_info(runs)")
        if row["name"] == "audio_sha256"
    )
    assert audio["notnull"] == 0
    connection.close()


# -- page ------------------------------------------------------------------


def test_realtime_page_renders(client):
    body = client.get("/realtime").text
    assert "Realtime transcription" in body
    assert 'id="rt-record"' in body
    assert 'id="rt-cadence"' in body
    assert "rt-provider" in body
    # Both models are offered, and Phonon's word-timestamp gap is declared.
    assert "Whistle" in body
    assert "Whistle only, live" in body


def test_realtime_page_explains_the_cadence_tradeoff(client):
    body = client.get("/realtime").text
    assert "1.07" in body and "0.63" in body


def test_navigation_links_both_pages(client):
    assert 'href="/realtime"' in client.get("/").text
    assert 'href="/"' in client.get("/realtime").text
