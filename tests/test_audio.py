"""Audio normalization."""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.prepare import load_samples, prepare_audio
from app.errors import AudioError


def test_resamples_and_downmixes_to_16k_mono(settings, synth_stereo_44k):
    prepared = prepare_audio(synth_stereo_44k, settings.audio_dir)
    assert prepared.sample_rate == 16000
    assert prepared.channels == 1
    assert prepared.duration_ms == pytest.approx(500, abs=10)
    assert prepared.path.exists()


def test_normalized_audio_reads_back_as_float32(settings, synth_stereo_44k):
    prepared = prepare_audio(synth_stereo_44k, settings.audio_dir)
    samples = load_samples(prepared.path, prepared.sample_rate)
    assert samples.dtype == np.float32
    assert len(samples) == pytest.approx(8000, abs=16)


def test_hash_is_stable_and_content_addressed(settings, synth_wav):
    first = prepare_audio(synth_wav, settings.audio_dir)
    second = prepare_audio(synth_wav, settings.audio_dir)
    assert first.sha256 == second.sha256
    # Identical audio resolves to the same cached file, so providers share bytes.
    assert first.path == second.path


def test_different_audio_hashes_differently(settings, synth_wav, synth_stereo_44k):
    first = prepare_audio(synth_wav, settings.audio_dir)
    second = prepare_audio(synth_stereo_44k, settings.audio_dir)
    assert first.sha256 != second.sha256


def test_hash_ignores_the_source_container(settings, synth_wav, tmp_path):
    """The same samples written twice hash the same, name and container aside."""
    import soundfile as sf

    samples, rate = sf.read(str(synth_wav), dtype="float32")
    copy = tmp_path / "identical-different-name.wav"
    sf.write(str(copy), samples, rate, subtype="PCM_16")

    assert (
        prepare_audio(synth_wav, settings.audio_dir).sha256
        == prepare_audio(copy, settings.audio_dir).sha256
    )


def test_undecodable_file_raises_audio_error(settings, tmp_path):
    broken = tmp_path / "not-really.wav"
    broken.write_text("this is plainly not audio", encoding="utf-8")
    with pytest.raises(AudioError) as excinfo:
        prepare_audio(broken, settings.audio_dir)
    assert "could not decode" in str(excinfo.value)


def test_missing_file_raises_audio_error(settings, tmp_path):
    with pytest.raises(AudioError) as excinfo:
        prepare_audio(tmp_path / "absent.wav", settings.audio_dir)
    assert "not found" in str(excinfo.value)


def test_empty_audio_is_rejected(settings, tmp_path):
    import soundfile as sf

    empty = tmp_path / "empty.wav"
    sf.write(str(empty), np.zeros(0, dtype=np.float32), 16000)
    with pytest.raises(AudioError):
        prepare_audio(empty, settings.audio_dir)


def test_browser_rate_mono_audio_is_accepted(settings, tmp_path):
    """An AudioContext usually runs at 48 kHz; the server must still cope.

    The recorder resamples to 16 kHz in the browser, but a recording that
    bypasses that path, or a fixture recorded elsewhere, must not be rejected.
    """
    import soundfile as sf

    rate = 48000
    seconds = 1.5
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    samples = (0.25 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    path = tmp_path / "recording-at-48k.wav"
    sf.write(str(path), samples, rate, subtype="PCM_16")

    prepared = prepare_audio(path, settings.audio_dir)
    assert prepared.sample_rate == 16000
    assert prepared.channels == 1
    assert prepared.duration_ms == pytest.approx(1500, abs=10)

    normalized = load_samples(prepared.path, prepared.sample_rate)
    assert len(normalized) == pytest.approx(24000, abs=200)


def test_pcm16_recording_is_not_silently_altered(settings, tmp_path):
    """A 16 kHz mono PCM16 file, exactly what the browser recorder emits.

    Re-normalizing an already-normal file must be a pass-through: same rate,
    same length, and stable across repeated preparation.
    """
    import soundfile as sf

    rate = 16000
    frames = 8000  # 0.5 s
    t = np.linspace(0, 0.5, frames, endpoint=False)
    samples = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "recording-2026-01-01T00-00-00.wav"
    sf.write(str(path), samples, rate, subtype="PCM_16")

    first = prepare_audio(path, settings.audio_dir)
    assert first.sample_rate == 16000
    assert first.duration_ms == pytest.approx(500, abs=1)

    second = prepare_audio(path, settings.audio_dir)
    assert second.sha256 == first.sha256
    assert second.path == first.path
