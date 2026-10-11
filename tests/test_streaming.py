"""Streaming events, metrics, protocol framing and the rolling segmenter.

The segmenter is the fallback for providers without a streaming API, so it is
tested against an injected decoder: no model is needed, and the behaviour that
matters (voice gating, closing a segment, never dropping audio) is checked
directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.streaming.events import (
    StreamEvent,
    StreamKind,
    StreamMetrics,
    StreamResult,
    StreamState,
)
from app.streaming.segmenter import RollingSegmenter
from app.workers import protocol

RATE = 16000


def tone(seconds: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * RATE), dtype=np.float32) / RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * RATE), dtype=np.float32)


# -- events ----------------------------------------------------------------


def test_partial_round_trips_through_the_wire():
    event = StreamEvent.partial("hello there", pending="wor", received_s=1.25, pass_ms=88.4)
    payload = event.to_wire()
    assert payload["type"] == "partial"
    assert payload["text"] == "hello there"
    assert payload["pending"] == "wor"

    restored = StreamEvent.from_wire(payload)
    assert restored.kind is StreamKind.PARTIAL
    assert restored.text == "hello there"
    assert restored.pending == "wor"
    assert restored.received_s == 1.25


def test_final_carries_its_segment_number():
    restored = StreamEvent.from_wire(
        StreamEvent.final("a phrase", segment=3).to_wire()
    )
    assert restored.segment == 3
    assert restored.kind is StreamKind.FINAL


def test_status_states_round_trip():
    for state in StreamState:
        restored = StreamEvent.from_wire(
            StreamEvent.status(state, detail="why").to_wire()
        )
        assert restored.state is state
        assert restored.detail == "why"


def test_error_round_trips():
    restored = StreamEvent.from_wire(StreamEvent.error("engine busy").to_wire())
    assert restored.kind is StreamKind.ERROR
    assert restored.message == "engine busy"


def test_missing_measurements_are_omitted_not_zeroed():
    """Zero is a claim; absent is the truth."""
    payload = StreamEvent.partial("text").to_wire()
    assert "passMs" not in payload
    assert "receivedS" not in payload


def test_metrics_never_invent_a_realtime_factor():
    metrics = StreamMetrics()
    assert metrics.realtime_factor is None
    assert metrics.as_dict()["realtime_factor"] is None


def test_metrics_running_mean_and_realtime_factor():
    metrics = StreamMetrics(audio_s=10.0)
    metrics.observe_pass(100.0)
    metrics.observe_pass(300.0)
    assert metrics.decodes == 2
    assert metrics.mean_pass_ms == pytest.approx(200.0)
    assert metrics.max_pass_ms == 300.0
    # 2 decodes of 200 ms over 10 s of audio.
    assert metrics.realtime_factor == pytest.approx(0.04)


def test_result_reports_error_and_implementation():
    result = StreamResult(provider="x", text="hi", error="cut short",
                          implementation="rolling")
    assert result.ok is False
    assert result.as_dict()["implementation"] == "rolling"


# -- protocol framing ------------------------------------------------------


def test_audio_block_round_trips_exactly():
    samples = tone(0.25)
    encoded = protocol.encode_audio_block(samples)
    assert isinstance(encoded, str)
    restored = protocol.decode_audio_block(encoded)
    assert restored.dtype == np.float32
    assert np.array_equal(restored, samples)


def test_stream_options_round_trip():
    from app.streaming.session import StreamOptions

    original = StreamOptions(
        language="de", keywords=("Ada", "Quillon"), word_timestamps=True,
        chunk_seconds=2.0,
    )
    restored = StreamOptions.from_wire(original.as_wire())
    assert restored == original


def test_stream_options_tolerate_a_missing_cadence():
    from app.streaming.session import StreamOptions

    assert StreamOptions.from_wire({}).chunk_seconds is None


def test_job_request_wire_shape_is_unchanged():
    """The streaming addition must not disturb the batch contract."""
    request = protocol.JobRequest(
        job_id="abc",
        audio=__import__("app.domain.models", fromlist=["PreparedAudio"]).PreparedAudio(
            path="x.wav", sha256="deadbeef", duration_ms=1000.0,
            sample_rate=16000, channels=1,
        ),
        options=__import__("app.domain.models", fromlist=["TranscriptionOptions"]).TranscriptionOptions(),
    )
    payload = request.to_wire()
    assert payload["type"] == protocol.JOB
    assert payload["audio"]["sha256"] == "deadbeef"
    assert protocol.JobRequest.from_wire(payload).job_id == "abc"


# -- rolling segmenter -----------------------------------------------------


class Recorder:
    """Injected decoder that remembers what it was asked to decode."""

    def __init__(self, text: str = "words") -> None:
        self.calls: list[int] = []
        self.text = text

    def __call__(self, audio: np.ndarray) -> str:
        self.calls.append(audio.size)
        return self.text


def test_first_partial_lands_after_the_configured_audio():
    decode = Recorder()
    segmenter = RollingSegmenter(decode, RATE)

    assert segmenter.feed(tone(0.2)) == []  # below the first-partial threshold
    events = segmenter.feed(tone(0.2))

    assert [event.kind for event in events] == [StreamKind.PARTIAL]
    assert events[0].text == "words"


def test_silence_alone_never_produces_text():
    """Room tone must not be hallucinated into a transcript."""
    decode = Recorder()
    segmenter = RollingSegmenter(decode, RATE)

    assert segmenter.feed(silence(3.0)) == []
    assert decode.calls == []
    text, events = segmenter.finish()
    assert text == ""
    assert events == []


def test_trailing_silence_closes_a_segment_and_emits_a_final():
    decode = Recorder("hello")
    segmenter = RollingSegmenter(decode, RATE)

    segmenter.feed(tone(1.0))
    events = segmenter.feed(silence(0.8))

    kinds = [event.kind for event in events]
    assert StreamKind.FINAL in kinds
    final = next(event for event in events if event.kind is StreamKind.FINAL)
    assert final.text == "hello"
    assert final.segment == 1
    assert segmenter.transcript == "hello"


def test_segments_accumulate_in_order():
    decode = Recorder("phrase")
    segmenter = RollingSegmenter(decode, RATE)

    for index in range(3):
        segmenter.feed(tone(0.8))
        segmenter.feed(silence(0.8))

    assert segmenter.finish()[0] == "phrase phrase phrase"
    assert segmenter.metrics.segments == 3


def test_the_final_decode_sees_all_audio_from_its_segment():
    """The cut includes the trailing quiet, so a final decode is gap-free."""
    decode = Recorder()
    segmenter = RollingSegmenter(decode, RATE)

    segmenter.feed(tone(1.0))
    segmenter.feed(silence(0.8))

    assert decode.calls, "the segment was never decoded"
    # The segment closes once trailing silence reaches its 0.70 s threshold, and
    # that quiet is part of what gets decoded: 1.0 s of speech plus 0.70 s.
    assert decode.calls[-1] == pytest.approx(1.70 * RATE, abs=RATE * 0.06)


def test_a_long_monologue_is_capped_rather_than_unbounded():
    decode = Recorder("talking")
    segmenter = RollingSegmenter(decode, RATE)

    # 35 s of continuous speech exceeds the 30 s segment cap.
    segmenter.feed(tone(35.0))

    assert segmenter.metrics.segments >= 1
    assert all(size <= 30.0 * RATE + RATE * 0.1 for size in decode.calls)


def test_idle_room_tone_is_trimmed_not_buffered_forever(monkeypatch):
    decode = Recorder()
    segmenter = RollingSegmenter(decode, RATE, max_segment_s=1.0)

    # Three seconds of pure room tone in a 1 s cap: the buffer must not grow.
    segmenter.feed(silence(3.0))

    assert decode.calls == []
    assert segmenter._samples <= RATE + 1


def test_finish_flushes_a_short_recording():
    """Whistle-style: a recording shorter than a segment still gets transcribed."""
    decode = Recorder("short take")
    segmenter = RollingSegmenter(decode, RATE)

    segmenter.feed(tone(0.5))
    text, events = segmenter.finish()

    assert text == "short take"
    assert [event.kind for event in events] == [StreamKind.FINAL]


def test_finish_is_idempotent():
    decode = Recorder("once")
    segmenter = RollingSegmenter(decode, RATE)

    segmenter.feed(tone(0.6))
    first = segmenter.finish()
    second = segmenter.finish()

    assert first[0] == second[0] == "once"


def test_a_failing_decode_still_reports_timing():
    def broken(audio):
        raise RuntimeError("engine exploded")

    segmenter = RollingSegmenter(broken, RATE)
    with pytest.raises(RuntimeError):
        segmenter.feed(tone(1.0))

    # The pass was still measured, so a failing provider reports honest timing.
    assert segmenter.metrics.decodes == 1


def test_self_clocking_prevents_queued_decodes():
    """Partial cadence is scheduled from the buffer after each decode."""
    decode = Recorder()
    segmenter = RollingSegmenter(decode, RATE)

    # 1.5 s of one burst: the first partial at 0.35 s, then one per 0.5 s.
    segmenter.feed(tone(1.5))

    assert 2 <= len(decode.calls) <= 4
