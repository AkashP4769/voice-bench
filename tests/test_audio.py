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
