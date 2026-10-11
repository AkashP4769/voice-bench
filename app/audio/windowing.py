"""Chunking for providers that cannot take long audio in one pass.

Both providers in this project segment long recordings themselves, so neither
routes through here today. The code exists because the capability is declared
per provider: a provider that reports ``max_duration_s`` will be chunked by
this module, and the strategy is recorded on the result because our windowing
is a confounder for word error rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.domain.models import Word


@dataclass(frozen=True)
class Window:
    index: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def plan_windows(
    duration_s: float,
    window_s: float,
    overlap_s: float = 0.0,
) -> list[Window]:
    """Cover ``duration_s`` with overlapping windows.

    The final window is clamped to the end of the audio rather than padded, so
    timestamps stay inside the recording.
    """
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    if overlap_s < 0:
        raise ValueError("overlap_s must not be negative")
    if overlap_s >= window_s:
        raise ValueError("overlap_s must be smaller than window_s")
    if duration_s <= 0:
        return []

    if duration_s <= window_s:
        return [Window(0, 0.0, duration_s)]

    step = window_s - overlap_s
    windows: list[Window] = []
    start = 0.0
    index = 0
    while start < duration_s:
        end = min(start + window_s, duration_s)
        windows.append(Window(index, start, end))
        if end >= duration_s:
            break
        start += step
        index += 1
    return windows


def slice_window(samples: np.ndarray, sample_rate: int, window: Window) -> np.ndarray:
    start = int(round(window.start_s * sample_rate))
    end = int(round(window.end_s * sample_rate))
    return np.ascontiguousarray(samples[start:end])


def to_absolute(offset_s: float, words: list[Word]) -> list[Word]:
    """Shift window-relative word times onto the timeline of the whole file."""
    offset_ms = offset_s * 1000.0
    return [
        Word(
            text=w.text,
            start_ms=w.start_ms + offset_ms,
            end_ms=w.end_ms + offset_ms,
            probability=w.probability,
        )
        for w in words
    ]


def assemble_words(
    pieces: list[tuple[Window, list[Word]]],
    overlap_s: float,
) -> list[Word]:
    """Combine per-window words into one absolute timeline.

    Words are relative to their own window, so each set is offset first. Every
    window except the last also re-decodes the trailing ``overlap_s`` of audio,
    and those words are dropped from the earlier window because the next window
    decodes them with right-context available.

    This is decided by position rather than by comparing word text: the same
    word stated once could otherwise be kept twice if the two windows disagreed
    slightly on where it started, and matching text across windows can also
    delete a genuinely repeated word.
    """
    assembled: list[Word] = []
    last_index = len(pieces) - 1
    for index, (window, words) in enumerate(pieces):
        cutoff_ms = None if index == last_index else (window.end_s - overlap_s) * 1000.0
        for word in to_absolute(window.start_s, words):
            if cutoff_ms is not None and word.start_ms >= cutoff_ms:
                continue
            assembled.append(word)
    assembled.sort(key=lambda word: (word.start_ms, word.end_ms))
    return assembled


def stitch_text(parts: list[str]) -> str:
    """Fallback for providers that cannot give word times.

    Without timestamps the overlap cannot be de-duplicated, so this is recorded
    as a distinct strategy rather than presented as equivalent to
    ``assemble_words``.
    """
    return " ".join(part.strip() for part in parts if part and part.strip())
