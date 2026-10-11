"""Chunking and stitching."""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.windowing import (
    assemble_words,
    plan_windows,
    slice_window,
    stitch_text,
    to_absolute,
)
from app.domain.models import Word


def test_short_audio_is_one_window():
    windows = plan_windows(10.0, window_s=28.0)
    assert len(windows) == 1
    assert windows[0].start_s == 0.0
    assert windows[0].end_s == 10.0


def test_long_audio_is_tiled_with_overlap():
    windows = plan_windows(60.0, window_s=28.0, overlap_s=1.0)
    assert [round(w.start_s) for w in windows] == [0, 27, 54]
    # The last window is clamped instead of padded, so timestamps stay in range.
    assert windows[-1].end_s == 60.0


def test_zero_duration_yields_no_windows():
    assert plan_windows(0.0, window_s=28.0) == []


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        plan_windows(10.0, window_s=0)
    with pytest.raises(ValueError):
        plan_windows(10.0, window_s=10.0, overlap_s=-1)
    with pytest.raises(ValueError):
        plan_windows(10.0, window_s=10.0, overlap_s=10.0)


def test_slice_window_returns_the_requested_samples():
    samples = np.arange(16000, dtype=np.float32)
    window = plan_windows(1.0, window_s=0.5)[0]
    chunk = slice_window(samples, 16000, window)
    assert chunk.size == 8000
    assert chunk[0] == 0
    assert chunk[-1] == 7999


def test_to_absolute_shifts_times():
    words = [Word(text="a", start_ms=100.0, end_ms=200.0, probability=None)]
    shifted = to_absolute(5.0, words)
    assert shifted[0].start_ms == 5100.0
    assert shifted[0].end_ms == 5200.0


def test_assemble_words_offsets_windows_onto_one_timeline():
    windows = plan_windows(55.0, window_s=28.0, overlap_s=1.0)
    assert [round(w.start_s) for w in windows] == [0, 27]

    pieces = [
        (windows[0], [Word(text="hello", start_ms=0.0, end_ms=400.0)]),
        (windows[1], [Word(text="world", start_ms=1000.0, end_ms=1400.0)]),
    ]
    words = assemble_words(pieces, overlap_s=1.0)
    assert [w.text for w in words] == ["hello", "world"]
    assert words[0].start_ms == 0.0
    assert words[1].start_ms == 28_000.0


def test_assemble_words_drops_the_earlier_copy_in_the_trailing_overlap():
    """The next window decodes the overlap with right-context, so it wins."""
    windows = plan_windows(55.0, window_s=28.0, overlap_s=1.0)
    first = [
        Word(text="kept", start_ms=26_000.0, end_ms=26_500.0),
        Word(text="dropped", start_ms=27_500.0, end_ms=28_000.0),
    ]
    second = [
        Word(text="dropped", start_ms=500.0, end_ms=1_000.0),
        Word(text="later", start_ms=1_000.0, end_ms=1_500.0),
    ]
    words = assemble_words([(windows[0], first), (windows[1], second)], overlap_s=1.0)

    assert [w.text for w in words] == ["kept", "dropped", "later"]
    # The surviving copy is the one offset from the second window.
    dropped = next(w for w in words if w.text == "dropped")
    assert dropped.start_ms == 27_500.0


def test_assemble_words_keeps_everything_from_the_final_window():
    windows = plan_windows(30.0, window_s=20.0, overlap_s=2.0)
    last = windows[-1]
    pieces = [(last, [Word(text="tail", start_ms=0.0, end_ms=300.0)])]
    words = assemble_words(pieces, overlap_s=2.0)
    assert [w.text for w in words] == ["tail"]


def test_assemble_words_keeps_a_genuinely_repeated_word():
    """De-duplication is positional, so a repeated word is never eaten."""
    windows = plan_windows(10.0, window_s=20.0, overlap_s=1.0)
    pieces = [
        (
            windows[0],
            [
                Word(text="no", start_ms=0.0, end_ms=200.0),
                Word(text="no", start_ms=900.0, end_ms=1_100.0),
            ],
        )
    ]
    words = assemble_words(pieces, overlap_s=1.0)
    assert [w.text for w in words] == ["no", "no"]


def test_stitch_text_joins_and_skips_blanks():
    assert stitch_text(["hello", "", "  ", "world"]) == "hello world"
