"""Per-provider live sessions running concurrently.

Each provider gets its own thread, because an engine decode blocks: running them
inline would serialize the models against each other and stall the event loop,
which defeats the point of comparing them live.

Audio is broadcast to every provider through a **bounded** buffer. A provider
that cannot keep up is stopped with an explicit error rather than having audio
silently dropped, because a transcript missing audio would look like a model
that misheard rather than a harness that gave up.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading

import numpy as np

from app.streaming.events import (
    StreamEvent,
    StreamMetrics,
    StreamResult,
    StreamState,
)
from app.streaming.session import StreamOptions

logger = logging.getLogger(__name__)

#: Audio a single slow provider may fall behind before its session is stopped.
BUFFER_SECONDS = 30.0
SAMPLE_RATE = 16000
LOOP_JOIN_TIMEOUT_S = 20.0


class _Lane:
    """One provider's thread, queue and buffer accounting."""

    def __init__(self, name: str, stream, manager: "StreamManager") -> None:
        self.name = name
        self.stream = stream
        self.manager = manager
        self.queue: queue.Queue = queue.Queue()
        self.max_buffer_samples = int(BUFFER_SECONDS * SAMPLE_RATE)
        self._buffered = 0
        self._lock = threading.Lock()
        self.result: StreamResult | None = None
        self.thread = threading.Thread(
            target=self._run, name=f"stream-{name}", daemon=True
        )

    # -- producer side (called from the websocket task) --------------------

    def offer(self, samples: np.ndarray) -> bool:
        """Queue audio, or report that this provider has fallen too far behind."""
        with self._lock:
            if self._buffered + samples.size > self.max_buffer_samples:
                return False
            self._buffered += samples.size
        self.queue.put(samples)
        return True

    def request_stop(self) -> None:
        self.queue.put(None)

    # -- consumer side -----------------------------------------------------

    def _run(self) -> None:
        result: StreamResult | None = None
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                with self._lock:
                    self._buffered -= item.size
                for event in self.stream.feed(item):
                    self.manager.emit(event, self.name)
            result = self.stream.stop()
        except Exception as exc:
            logger.warning("stream for %s failed: %s", self.name, exc)
            self.manager.emit(StreamEvent.error(str(exc)), self.name)
            metrics = getattr(self.stream, "metrics", None) or StreamMetrics()
            result = StreamResult(
                provider=self.name,
                metrics=metrics,
                error=str(exc),
                implementation=getattr(self.stream, "implementation", None),
            )
            # A failed feed already abandoned the session; only abort if not.
            if not getattr(self.stream, "finished", False):
                try:
                    self.stream.abort()
                except Exception:
                    logger.debug("abort for %s did not complete", self.name)
        finally:
            if result is not None:
                # Adapters leave `provider` empty because they do not know their
                # configured name; without this the transcript would be keyed
                # under an empty string in the final payload.
                if not result.provider:
                    result.provider = self.name
                self.result = result
                self.manager.publish_result(result)


class StreamManager:
    """Owns every lane for one realtime session."""

    def __init__(
        self,
        pool,
        providers: list[str],
        options: StreamOptions,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.pool = pool
        self.providers = providers
        self.options = options
        self._loop = loop
        self.out: asyncio.Queue = asyncio.Queue()
        self._lanes: dict[str, _Lane] = {}
        self._closed = False
        self._closing = False
        #: Providers that could not open a session, and the reason.
        self.failed: dict[str, str] = {}
        self.implementations: dict[str, str] = {}

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        for name in self.providers:
            try:
                stream = self.pool.open_stream(name, self.options)
            except Exception as exc:
                self.failed[name] = str(exc)
                self.emit(StreamEvent.error(str(exc)), name)
                continue

            self.implementations[name] = getattr(stream, "implementation", None)
            lane = _Lane(name, stream, self)
            self._lanes[name] = lane
            lane.thread.start()

            detail = None
            if self.options.word_timestamps and name != "whistle":
                detail = (
                    "word timestamps are not available for this provider in "
                    "realtime; the transcript is reported without them"
                )
            self.emit(
                StreamEvent.status(StreamState.STREAMING, detail=detail), name
            )

    def push(self, samples: np.ndarray) -> None:
        if self._closed or samples.size == 0:
            return
        for name, lane in self._lanes.items():
            if lane.result is not None:
                # Already finished (usually an error); nothing more to feed.
                continue
            if not lane.offer(samples):
                self._overrun(name, lane)

    def _overrun(self, name: str, lane: _Lane) -> None:
        message = (
            f"{name} fell more than {BUFFER_SECONDS:.0f}s behind and was stopped "
            "so audio would not be silently dropped. Its transcript is "
            "incomplete; try one model at a time, or a slower update cadence."
        )
        self.emit(StreamEvent.status(StreamState.OVERRUN, detail=message), name)
        self.emit(StreamEvent.error(message), name)
        metrics = getattr(lane.stream, "metrics", None) or StreamMetrics()
        metrics.overruns += 1
        lane.result = StreamResult(
            provider=name,
            text=_partial_text(lane),
            metrics=metrics,
            error=message,
            implementation=getattr(lane.stream, "implementation", None),
        )
        if not getattr(lane.stream, "finished", False):
            try:
                lane.stream.abort()
            except Exception:
                logger.debug("abort after overrun for %s did not complete", name)
        # Unblock the lane thread so it exits rather than waiting forever.
        lane.request_stop()
        self.publish_result(lane.result)

    def close(self) -> list[StreamResult]:
        """Stop every lane and return the results."""
        self._closed = True
        for lane in self._lanes.values():
            lane.request_stop()
        for lane in self._lanes.values():
            lane.thread.join(timeout=LOOP_JOIN_TIMEOUT_S)
            if lane.thread.is_alive():
                logger.warning("stream thread for %s did not exit", lane.name)
                if not getattr(lane.stream, "finished", False):
                    try:
                        lane.stream.abort()
                    except Exception:
                        logger.debug("late abort for %s failed", lane.name)

        results: list[StreamResult] = []
        for name in self.providers:
            lane = self._lanes.get(name)
            if lane is not None and lane.result is not None:
                results.append(lane.result)
            elif name in self.failed:
                results.append(
                    StreamResult(provider=name, error=self.failed[name])
                )
        return results

    # -- event plumbing ----------------------------------------------------

    def emit(self, event: StreamEvent, provider: str) -> None:
        event.provider = provider
        self._post(("event", event))

    def publish_result(self, result: StreamResult) -> None:
        if not result.provider:
            result.provider = "unknown"
        self._post(("result", result))

    def _post(self, item) -> None:
        if self._closing:
            return
        try:
            # Called from lane threads, so the handoff must be scheduled on the
            # loop rather than touching the asyncio queue directly.
            self._loop.call_soon_threadsafe(self.out.put_nowait, item)
        except RuntimeError:
            # The loop is gone, which means the socket already closed.
            logger.debug("dropping a stream message: event loop is closed")


def _partial_text(lane: _Lane) -> str:
    """Best-effort transcript for a session that was cut short."""
    text = getattr(lane.stream, "_committed", None)
    if isinstance(text, str) and text:
        return text
    best = getattr(lane.stream, "_best_text", None)
    if callable(best):
        try:
            return best()
        except Exception:
            return ""
    return ""
