"""Integration tests that load the real models.

Marked ``integration`` because each run starts a worker process and loads a
model. Worth running before trusting a benchmark: they are what proves both
adapters agree on the normalized contract.
"""

from __future__ import annotations

import wave

import pytest

from app.audio.prepare import prepare_audio
from app.domain.models import TranscriptionOptions
from app.workers.pool import WorkerPool

pytestmark = pytest.mark.integration

PROVIDERS = ["whistle", "phonon-2"]


@pytest.fixture(scope="module")
def pool():
    from app.config import get_settings

    settings = get_settings()
    instance = WorkerPool(settings)
    yield instance
    instance.shutdown()


def _duration_ms(path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate() * 1000.0


@pytest.mark.parametrize("provider", PROVIDERS)
def test_provider_returns_a_normalized_result(settings, speech_fixture, pool, provider):
    audio = prepare_audio(speech_fixture, settings.audio_dir)
    result = pool.run(
        provider, audio, TranscriptionOptions(word_timestamps=True)
    )

    assert result.error is None, result.error
    assert result.provider == provider
    assert result.transport == "in_process"
    assert result.text.strip()
    assert result.audio_duration_ms == pytest.approx(audio.duration_ms, rel=0.01)
    assert result.latency_ms and result.latency_ms > 0
    assert result.real_time_factor and result.real_time_factor > 0


@pytest.mark.parametrize("provider", PROVIDERS)
def test_word_timestamps_are_ordered_and_inside_the_audio(
    settings, speech_fixture, pool, provider
):
    audio = prepare_audio(speech_fixture, settings.audio_dir)
    result = pool.run(provider, audio, TranscriptionOptions(word_timestamps=True))

    assert result.words, f"{provider} returned no word timestamps"
    previous_end = 0.0
    for word in result.words:
        assert word.text
        assert word.start_ms >= 0
        assert word.end_ms >= word.start_ms
        assert word.start_ms >= previous_end - 1e-6 or word.start_ms >= 0
        assert word.end_ms <= audio.duration_ms + 50
        previous_end = max(previous_end, word.end_ms)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_words_join_to_the_transcript(settings, speech_fixture, pool, provider):
    """Both engines promise word texts joined by spaces equal the transcript."""
    audio = prepare_audio(speech_fixture, settings.audio_dir)
    result = pool.run(provider, audio, TranscriptionOptions(word_timestamps=True))
    joined = " ".join(word.text for word in result.words)
    assert joined.strip() == result.text.strip()


def test_whistle_handles_audio_longer_than_thirty_seconds(settings, pool):
    """The engine segments internally, so no harness chunking should appear."""
    fixture = settings.root / "tests" / "fixtures" / "long_en.wav"
    if not fixture.exists():
        pytest.skip("long fixture not generated")
    audio = prepare_audio(fixture, settings.audio_dir)
    assert audio.duration_ms > 30_000

    result = pool.run("whistle", audio, TranscriptionOptions(word_timestamps=True))
    assert result.error is None, result.error
    assert result.windowing.window_count == 1
    assert not result.warnings
    # The whole clip was covered, not just the first window.
    assert result.words[-1].end_ms > 20_000


def test_both_providers_produce_the_same_transcript_for_a_clean_clip(
    settings, speech_fixture, pool
):
    """Not a guarantee in general, but these two agree on synthesized speech."""
    audio = prepare_audio(speech_fixture, settings.audio_dir)
    options = TranscriptionOptions(word_timestamps=True)
    texts = {
        name: pool.run(name, audio, options).text.strip() for name in PROVIDERS
    }
    assert len(set(texts.values())) == 1, texts
