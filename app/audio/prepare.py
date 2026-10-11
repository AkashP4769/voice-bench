"""Audio normalization.

Every provider receives byte-identical, 16 kHz mono float32 audio. Both engines
expect 16 kHz mono, and normalizing once means:

* a comparison is never decided by one provider getting an easier input;
* the SHA-256 of the normalized samples identifies the exact audio that was
  scored, independent of the container the user uploaded.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from app.domain.models import PreparedAudio
from app.errors import AudioError

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    try:
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile raises several unrelated types
        raise AudioError(
            f"could not decode {path.name}: {exc}. "
            "Supported containers are those libsndfile reads (wav, flac, ogg, aiff). "
            "Convert the file first, for example: ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav"
        ) from exc

    if data.size == 0:
        raise AudioError(f"{path.name} contains no audio samples")

    if data.shape[1] > TARGET_CHANNELS:
        mono = data.mean(axis=1)
    else:
        mono = data[:, 0]
    return np.ascontiguousarray(mono, dtype=np.float32), sample_rate


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples
    resampled = soxr.resample(samples, source_rate, target_rate, quality="HQ")
    return np.ascontiguousarray(resampled, dtype=np.float32)


def fingerprint(samples: np.ndarray, sample_rate: int) -> str:
    """Hash the normalized samples, not the file, so containers do not matter."""
    digest = hashlib.sha256()
    digest.update(f"{sample_rate}:{samples.shape[0]}".encode("ascii"))
    digest.update(samples.tobytes())
    return digest.hexdigest()


def prepare_audio(
    source: Path,
    cache_dir: Path,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> PreparedAudio:
    """Normalize ``source`` to 16 kHz mono and cache the result.

    Returns the cached path plus the identity of the audio that was normalized.
    """
    source = Path(source)
    if not source.exists():
        raise AudioError(f"audio file not found: {source}")

    samples, sample_rate = _read_mono(source)
    samples = _resample(samples, sample_rate, target_sample_rate)

    sha256 = fingerprint(samples, target_sample_rate)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized = cache_dir / f"{sha256[:16]}.wav"

    if not normalized.exists():
        # PCM_16 keeps the file readable by any consumer; the float32 array is
        # what gets hashed and what providers with an in-process path receive.
        sf.write(str(normalized), samples, target_sample_rate, subtype="PCM_16")

    return PreparedAudio(
        path=normalized,
        sha256=sha256,
        duration_ms=len(samples) / target_sample_rate * 1000.0,
        sample_rate=target_sample_rate,
        channels=TARGET_CHANNELS,
    )


def load_samples(path: Path, expected_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Read normalized audio back as mono float32 for in-process providers."""
    samples, sample_rate = _read_mono(Path(path))
    if sample_rate != expected_rate:
        raise AudioError(
            f"expected {expected_rate} Hz audio at {path}, found {sample_rate} Hz"
        )
    return samples
