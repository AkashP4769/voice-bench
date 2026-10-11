"""Rolling-window segmentation for providers without a streaming API.

Both engines in this project stream natively, so this is the safety net, not the
main path:

* it is the fallback when Fermion's private ``LiveSession`` cannot be imported
  (see ``app/providers/phonon.py``);
* it makes any future provider work in realtime without a streaming API.

It re-decodes the whole in-flight segment rather than only the new audio, which
is what a stateless ``transcribe`` call can do, and it is bounded by
``max_segment_s`` so one unbroken monologue cannot grow an unbounded buffer or an
unbounded re-decode.

The cadence values mirror Fermion's own live session so the fallback behaves
similarly to the thing it replaces.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from app.streaming.events import StreamEvent, StreamMetrics

# Audio in the segment before the first partial decode.
FIRST_PARTIAL_S = 0.35
# Audio added between partial decodes after the first.
PARTIAL_EVERY_S = 0.50
# Sub-threshold trailing audio that closes a segment.
SILENCE_CLOSE_S = 0.70
# A segment closes at this length regardless, bounding the re-decode cost.
MAX_SEGMENT_S = 30.0
# Feed granularity, so the voice gate is evaluated at a sensible resolution.
BLOCK_S = 0.05
# RMS at or below this is room tone...
NOISE_FLOOR_RMS = 0.004
# ...as is anything this far below the segment's own peak.
GATE_RATIO = 0.18

DecodeFn = Callable[[np.ndarray], str]


class RollingSegmenter:
    """Turns a stream of blocks into partial and final events.

    ``decode`` receives 16 kHz mono float32 samples and returns text. It is
    injected so this class is testable without a model, and so the same
    segmentation serves any provider.
    """

    def __init__(
        self,
        decode: DecodeFn,
        sample_rate: int = 16000,
        metrics: StreamMetrics | None = None,
        *,
        first_partial_s: float = FIRST_PARTIAL_S,
        partial_every_s: float = PARTIAL_EVERY_S,
        silence_close_s: float = SILENCE_CLOSE_S,
        max_segment_s: float = MAX_SEGMENT_S,
        noise_floor_rms: float = NOISE_FLOOR_RMS,
        gate_ratio: float = GATE_RATIO,
    ) -> None:
        self._decode = decode
        self.sample_rate = sample_rate
        self.metrics = metrics if metrics is not None else StreamMetrics()
        self.first_partial_s = first_partial_s
        self.partial_every_s = partial_every_s
        self.silence_close_s = silence_close_s
        self.max_segment_s = max_segment_s
        self.noise_floor_rms = noise_floor_rms
        self.gate_ratio = gate_ratio
        self._finals: list[str] = []
        self._block_samples = max(1, int(BLOCK_S * sample_rate))
        self._reset_segment()

    # -- state -------------------------------------------------------------

    def _reset_segment(self) -> None:
        self._blocks: list[np.ndarray] = []
        self._samples = 0
        self._voiced = False
        self._last_voice = 0
        self._peak_rms = 0.0
        self._next_partial = int(self.first_partial_s * self.sample_rate)

    @property
    def transcript(self) -> str:
        return " ".join(self._finals)

    @property
    def finals(self) -> list[str]:
        return list(self._finals)

    # -- feeding -----------------------------------------------------------

    def feed(self, samples: np.ndarray) -> list[StreamEvent]:
        """Accept audio and return whatever it produced.

        Blocks are processed one gate-resolution step at a time, so the voice
        gate is not fooled by a chunk that is half speech and half silence.
        """
        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return []

        self.metrics.chunks += 1
        self.metrics.audio_s += block.size / self.sample_rate

        events: list[StreamEvent] = []
        for start in range(0, block.size, self._block_samples):
            events.extend(self._feed_block(block[start : start + self._block_samples]))
        return events

    def _feed_block(self, block: np.ndarray) -> list[StreamEvent]:
        self._blocks.append(block)
        self._samples += block.size

        rms = float(np.sqrt(np.mean(np.square(block.astype(np.float64)))))
        self._peak_rms = max(self._peak_rms, rms)
        gate = max(self.noise_floor_rms, self.gate_ratio * self._peak_rms)
        if rms > gate:
            self._voiced = True
            self._last_voice = self._samples

        trailing = self._samples - self._last_voice

        if self._voiced and trailing >= int(self.silence_close_s * self.sample_rate):
            return self._finalize()

        if self._samples >= int(self.max_segment_s * self.sample_rate):
            if self._voiced:
                return self._finalize()
            # An idle microphone: holding 30 s of room tone forever is the only
            # unbounded cost here. Keep the last second so a word starting on the
            # boundary keeps its onset. This is the one place audio is discarded,
            # and it is audio the gate already decided contains no speech.
            keep = self._concatenated()[-self.sample_rate :]
            self._reset_segment()
            self._blocks = [keep]
            self._samples = keep.size
            return []

        if self._voiced and self._samples >= self._next_partial:
            text = self._run_decode(self._concatenated())
            # Self-clock off the buffer as it stands *after* the decode, so a
            # slow re-decode cannot queue up further decodes behind it.
            self._next_partial = self._samples + int(
                self.partial_every_s * self.sample_rate
            )
            if text:
                return [StreamEvent.partial(text, received_s=self._received_s)]
        return []

    def _finalize(self) -> list[StreamEvent]:
        audio = self._concatenated()
        self._reset_segment()
        text = self._run_decode(audio)
        if not text:
            return []
        self._finals.append(text)
        self.metrics.segments += 1
        return [
            StreamEvent.final(
                text,
                segment=len(self._finals),
                received_s=self._received_s,
            )
        ]

    def _concatenated(self) -> np.ndarray:
        if not self._blocks:
            return np.zeros(0, dtype=np.float32)
        if len(self._blocks) == 1:
            return self._blocks[0]
        return np.concatenate(self._blocks)

    @property
    def _received_s(self) -> float:
        return self.metrics.audio_s

    def _run_decode(self, audio: np.ndarray) -> str:
        started = time.perf_counter()
        try:
            text = self._decode(audio)
        finally:
            # Recorded even when the decode raises, so a failing provider still
            # reports honest timing instead of nothing.
            self.metrics.observe_pass((time.perf_counter() - started) * 1000.0)
        return (text or "").strip()

    # -- finishing ---------------------------------------------------------

    def finish(self) -> tuple[str, list[StreamEvent]]:
        """Close the in-flight segment and return the full transcript.

        Audio buffered but never voiced is dropped here, which is correct: the
        gate already decided it contains no speech, and decoding room tone is
        exactly the hallucination both engines go out of their way to avoid.
        """
        events: list[StreamEvent] = []
        if self._samples and self._voiced:
            events = self._finalize()
        else:
            self._reset_segment()
        return self.transcript, events
