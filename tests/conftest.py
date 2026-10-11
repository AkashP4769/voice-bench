"""Shared test fixtures.

Unit tests build their own synthetic audio so they do not depend on the speech
fixtures, which are only needed by the integration tests that actually load a
model.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings, load_settings

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Real provider configuration, redirected to a temporary data directory."""
    base = load_settings(ROOT)
    return dataclasses.replace(
        base,
        data_dir=tmp_path,
        audio_dir=tmp_path / "audio",
        results_dir=tmp_path / "results",
        db_path=tmp_path / "bench.sqlite3",
    )


@pytest.fixture
def repository(settings: Settings):
    from app.storage.repository import Repository

    return Repository(settings.db_path)


@pytest.fixture
def synth_wav(tmp_path: Path) -> Path:
    """One second of 16 kHz mono speech-like noise."""
    generator = np.random.default_rng(seed=7)
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * 180 * t)
    samples = (tone + 0.02 * generator.standard_normal(t.size)).astype(np.float32)
    path = tmp_path / "synth.wav"
    sf.write(str(path), samples, 16000, subtype="PCM_16")
    return path


@pytest.fixture
def synth_stereo_44k(tmp_path: Path) -> Path:
    """Stereo 44.1 kHz, to prove normalization actually resamples and downmixes."""
    t = np.linspace(0, 0.5, 22050, endpoint=False)
    left = 0.3 * np.sin(2 * np.pi * 220 * t)
    right = 0.3 * np.sin(2 * np.pi * 330 * t)
    stereo = np.stack([left, right], axis=1).astype(np.float32)
    path = tmp_path / "stereo44.wav"
    sf.write(str(path), stereo, 44100, subtype="PCM_16")
    return path


@pytest.fixture
def speech_fixture() -> Path:
    path = FIXTURES / "short_en.wav"
    if not path.exists():
        pytest.skip(
            "speech fixtures are absent; run the fixture generator to create "
            f"{path}"
        )
    return path


class FakeWorkerPool:
    """Stands in for real workers so API tests never load a model."""

    def __init__(self, results: dict | None = None, status: list | None = None) -> None:
        self.results = results or {}
        self.status_data = status or []
        self.calls: list[tuple[str, object]] = []

    def run(self, name, audio, options):
        from app.domain.models import TranscriptionResult

        self.calls.append((name, options))
        payload = self.results.get(name)
        if payload is None:
            return TranscriptionResult(
                provider=name,
                model=name,
                transport="in_process",
                text="",
                audio_duration_ms=audio.duration_ms,
                latency_ms=10.0,
                real_time_factor=0.01,
            )
        if isinstance(payload, Exception):
            raise payload
        return payload

    def warm(self, names):
        return {name: "ready" for name in names}

    def worker(self, name):
        raise NotImplementedError

    def status(self):
        return self.status_data

    def shutdown(self):
        pass


@pytest.fixture
def client(settings: Settings, repository):
    from fastapi.testclient import TestClient

    from app.api import deps
    from app.main import create_app

    app = create_app()
    fake_pool = FakeWorkerPool()
    app.dependency_overrides[deps.get_config] = lambda: settings
    app.dependency_overrides[deps.get_repository] = lambda: repository
    app.dependency_overrides[deps.get_worker_pool] = lambda: fake_pool
    with TestClient(app) as test_client:
        test_client.fake_pool = fake_pool
        yield test_client
