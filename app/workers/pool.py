"""Worker pool: one process per provider, one request at a time.

Why processes rather than threads: both engines allow a single model per process
and are not thread-safe, so loading two models in the API process would violate
both engines' documented constraint. A process per provider also isolates the
heavy torch import and contains a crash.

Why a lock per worker: Phonon's server queues concurrent requests behind one
lock and Whistle is explicitly not thread-safe. Serialising here means a
comparison is never distorted by two decodes contending for the same cores, and
the reported latency belongs to the model rather than to the queue.

The control channel is a loopback socket, and the worker's stdout and stderr go
to a log file, so engine progress output can never be mistaken for protocol.
"""

from __future__ import annotations

import atexit
import json
import logging
import queue
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings
from app.domain.models import (
    PreparedAudio,
    TranscriptionOptions,
    TranscriptionResult,
)
from app.errors import ProviderError, ProviderUnavailable, ProviderTimeout
from app.streaming.events import StreamEvent, StreamResult
from app.streaming.session import StreamOptions
from app.workers import protocol

logger = logging.getLogger(__name__)

# The first run downloads and unpacks weights and compiles the runtime.
STARTUP_TIMEOUT_S = 900.0
JOB_TIMEOUT_S = 900.0
ACCEPT_POLL_S = 0.5


@dataclass
class WorkerStatus:
    provider: str
    running: bool = False
    load_seconds: float | None = None
    describe: dict = field(default_factory=dict)
    last_error: str | None = None
    restarts: int = 0

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "running": self.running,
            "load_seconds": self.load_seconds,
            "describe": self.describe,
            "last_error": self.last_error,
            "restarts": self.restarts,
        }


class ProviderWorker:
    """A single long-lived provider process."""

    def __init__(
        self,
        name: str,
        settings: Settings,
        startup_timeout_s: float = STARTUP_TIMEOUT_S,
        job_timeout_s: float = JOB_TIMEOUT_S,
    ) -> None:
        self.name = name
        self.settings = settings
        self.startup_timeout_s = startup_timeout_s
        self.job_timeout_s = job_timeout_s
        self.status = WorkerStatus(provider=name)
        self.log_path = settings.data_dir / "logs" / f"worker-{name}.log"
        self._process: subprocess.Popen | None = None
        self._conn: socket.socket | None = None
        self._writer = None
        self._log_handle = None
        self._responses: queue.Queue[dict] = queue.Queue()
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._started_once = False
        # True while a live session owns this provider. Read when building the
        # "busy" message, so a caller learns *why* it is refused.
        self._stream_active = False

    # -- lifecycle ---------------------------------------------------------

    def _spawn(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8", errors="replace")
        self._log_handle.write(
            f"\n===== starting worker {self.name} at {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"
        )
        self._log_handle.flush()

        command = [
            sys.executable,
            "-m",
            "app.workers.runner",
            self.name,
            str(port),
            str(self.settings.root),
        ]
        logger.info("starting worker for %s on port %s", self.name, port)
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(self.settings.root),
        )

        # Accept while watching the child, so a crash during import reports the
        # real reason instead of waiting out the startup timeout.
        deadline = time.monotonic() + self.startup_timeout_s
        listener.settimeout(ACCEPT_POLL_S)
        conn = None
        while True:
            try:
                conn, _ = listener.accept()
                break
            except socket.timeout:
                pass
            if self._process.poll() is not None:
                listener.close()
                raise ProviderError(self._startup_failure_message())
            if time.monotonic() > deadline:
                listener.close()
                self._terminate()
                raise ProviderTimeout(
                    f"{self.name} did not connect within {self.startup_timeout_s:g}s"
                )
        listener.close()

        conn.settimeout(None)
        self._conn = conn
        self._writer = conn.makefile("w", encoding="utf-8", newline="\n")
        self._responses = queue.Queue()
        self._reader = threading.Thread(
            target=self._read_loop, name=f"worker-{self.name}", daemon=True
        )
        self._reader.start()

    def _startup_failure_message(self) -> str:
        tail = self._log_tail()
        message = f"worker for {self.name} exited during startup"
        if tail:
            message += f". Last output from {self.log_path.name}:\n{tail}"
        return message

    def _log_tail(self, lines: int = 15) -> str:
        try:
            content = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(content.strip().splitlines()[-lines:])

    def _read_loop(self) -> None:
        conn = self._conn
        if conn is None:
            self._responses.put({"type": "eof"})
            return
        try:
            reader = conn.makefile("r", encoding="utf-8", newline="\n")
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._responses.put(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("non-protocol line from worker %s", self.name)
        except OSError:
            pass
        finally:
            self._responses.put({"type": "eof"})

    def _await(self, timeout_s: float, what: str) -> dict:
        try:
            return self._responses.get(timeout=timeout_s)
        except queue.Empty:
            self.status.last_error = f"timed out waiting for {what}"
            self._terminate()
            raise ProviderTimeout(
                f"{self.name} did not finish {what} within {timeout_s:g}s"
            ) from None

    def _terminate(self) -> None:
        process = self._process
        self._process = None
        self.status.running = False
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
        self._writer = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None

    def stop(self) -> None:
        with self._lock:
            process = self._process
            if process is not None and process.poll() is None and self._writer is not None:
                try:
                    self._writer.write(json.dumps({"type": protocol.SHUTDOWN}) + "\n")
                    self._writer.flush()
                except (OSError, ValueError):
                    pass
            self._terminate()

    def _handshake(self) -> None:
        message = self._await(self.startup_timeout_s, "model load")
        kind = message.get("type")
        if kind == protocol.FATAL:
            text = message.get("error", "worker failed to start")
            error_type = message.get("error_type", "")
            self.status.last_error = text
            self._terminate()
            if error_type in {ProviderUnavailable.__name__, ProviderError.__name__}:
                raise ProviderUnavailable(text)
            raise ProviderError(text)
        if kind != protocol.READY:
            self._terminate()
            raise ProviderError(
                f"worker for {self.name} sent {kind!r} instead of ready"
            )
        self.status.running = True
        self.status.load_seconds = message.get("load_seconds")
        self.status.describe = message.get("describe") or {}
        logger.info(
            "worker %s ready in %.1fs", self.name, self.status.load_seconds or 0.0
        )

    def start(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            self._spawn()
            self._started_once = True
            self._handshake()

    # -- work --------------------------------------------------------------

    def _ensure_running(self) -> None:
        """Start the worker if it is not up. Caller holds the lock."""
        if self._process is not None and self._process.poll() is None:
            return
        if self._started_once:
            self.status.restarts += 1
        self._terminate()
        self.status = WorkerStatus(provider=self.name, restarts=self.status.restarts)
        self._spawn()
        self._started_once = True
        self._handshake()

    def _busy_message(self) -> str:
        if self._stream_active:
            return (
                f"{self.name} has a live session open. Batch transcription is "
                "unavailable until that session ends, because the engine keeps "
                "stream state per process."
            )
        return f"{self.name} is busy with another request. Try again shortly."

    def _write(self, message: dict) -> None:
        if self._writer is None:
            raise ProviderError(f"worker for {self.name} is not running")
        try:
            self._writer.write(json.dumps(message) + "\n")
            self._writer.flush()
        except (OSError, ValueError) as exc:
            self._terminate()
            raise ProviderError(
                f"worker for {self.name} closed while sending work: {exc}"
            ) from exc

    def _await_reply(self, expected_id: str, timeout_s: float, what: str) -> dict:
        message = self._await(timeout_s, what)
        kind = message.get("type")
        if kind == "eof":
            self._terminate()
            raise ProviderError(
                f"worker for {self.name} exited during {what}. "
                f"Last output from {self.log_path.name}:\n{self._log_tail()}"
            )
        if message.get("id") != expected_id:
            self._terminate()
            raise ProviderError(
                f"worker for {self.name} sent {kind!r} for a different request"
            )
        return message

    @staticmethod
    def _raise_from(message: dict, provider: str) -> None:
        error_type = message.get("error_type", "")
        text = message.get("error", "provider failed")
        if error_type == ProviderTimeout.__name__:
            raise ProviderTimeout(text)
        if error_type in {ProviderUnavailable.__name__, ProviderError.__name__}:
            raise ProviderError(text)
        raise ProviderError(f"{provider} failed: {text}")

    def transcribe(
        self,
        audio: PreparedAudio,
        options: TranscriptionOptions,
    ) -> TranscriptionResult:
        # Non-blocking: a batch run behind a live session would otherwise hang
        # for as long as the user keeps talking, with no way to tell why.
        if not self._lock.acquire(blocking=False):
            raise ProviderError(self._busy_message())
        try:
            self._ensure_running()

            job_id = uuid.uuid4().hex
            request = protocol.JobRequest(job_id=job_id, audio=audio, options=options)
            self._write(request.to_wire())

            message = self._await_reply(job_id, self.job_timeout_s, "transcription")
            if message.get("type") != protocol.RESULT:
                self._terminate()
                raise ProviderError(
                    f"worker for {self.name} sent an unexpected message: "
                    f"{message.get('type')!r}"
                )
            if not message.get("ok"):
                self.status.last_error = message.get("error", "provider failed")
                self._raise_from(message, self.name)
            return TranscriptionResult.from_dict(message["result"])
        finally:
            self._lock.release()

    # -- streaming ---------------------------------------------------------

    def open_stream(
        self,
        options: StreamOptions,
        timeout_s: float | None = None,
    ) -> "ProviderStream":
        """Open a live session, holding this provider's lock for its lifetime.

        Fails fast rather than queueing: a live session is a user-facing
        activity, and a request waiting behind one could wait for minutes.
        """
        if not self._lock.acquire(blocking=False):
            raise ProviderError(self._busy_message())
        try:
            self._ensure_running()
            session_id = uuid.uuid4().hex
            self._write(
                {
                    "type": protocol.STREAM_START,
                    "id": session_id,
                    "options": options.as_wire(),
                }
            )
            message = self._await_reply(
                session_id, timeout_s or self.startup_timeout_s, "session start"
            )
            if message.get("type") == protocol.STREAM_ERROR:
                self.status.last_error = message.get("error", "stream failed")
                self._raise_from(message, self.name)
            if message.get("type") != protocol.STREAM_STARTED:
                self._terminate()
                raise ProviderError(
                    f"worker for {self.name} sent {message.get('type')!r} instead "
                    "of stream_started"
                )
            self._stream_active = True
            return ProviderStream(
                worker=self,
                session_id=session_id,
                implementation=message.get("implementation"),
            )
        except Exception:
            self._lock.release()
            raise

    def stream_feed(self, session_id: str, samples) -> list[StreamEvent]:
        """Caller holds the lock, via the open ProviderStream."""
        self._write(
            {
                "type": protocol.STREAM_CHUNK,
                "id": session_id,
                "audio": protocol.encode_audio_block(samples),
            }
        )
        message = self._await_reply(session_id, self.job_timeout_s, "stream chunk")
        kind = message.get("type")
        if kind == protocol.STREAM_ERROR:
            self.status.last_error = message.get("error", "stream failed")
            self._raise_from(message, self.name)
        if kind != protocol.STREAM_EVENTS:
            self._terminate()
            raise ProviderError(
                f"worker for {self.name} sent {kind!r} instead of stream_events"
            )
        return [StreamEvent.from_wire(event) for event in message.get("events") or []]

    def stream_close(self, session_id: str) -> StreamResult:
        """Caller holds the lock, via the open ProviderStream."""
        self._write({"type": protocol.STREAM_STOP, "id": session_id})
        message = self._await_reply(session_id, self.job_timeout_s, "session stop")
        kind = message.get("type")
        if kind == protocol.STREAM_ERROR:
            self.status.last_error = message.get("error", "stream failed")
            self._raise_from(message, self.name)
        if kind != protocol.STREAM_DONE:
            self._terminate()
            raise ProviderError(
                f"worker for {self.name} sent {kind!r} instead of stream_done"
            )
        return StreamResult(**message["result"])

    def release_stream(self) -> None:
        """Called once, when a ProviderStream finishes or is abandoned."""
        self._stream_active = False
        try:
            if self._lock.locked():
                self._lock.release()
        except RuntimeError:
            # Not held by this thread; nothing to release.
            logger.debug("stream lock for %s was already released", self.name)


class ProviderStream:
    """A live session handle that owns its provider's lock.

    Finishing is idempotent and happens even on failure, because a socket that
    drops mid-session must not leave the provider permanently busy.
    """

    def __init__(
        self,
        worker: ProviderWorker,
        session_id: str,
        implementation: str | None = None,
    ) -> None:
        self._worker = worker
        self.session_id = session_id
        self.implementation = implementation
        self.provider = worker.name
        self._finished = False

    def feed(self, samples) -> list[StreamEvent]:
        if self._finished:
            raise ProviderError(f"session for {self.provider} is already closed")
        try:
            return self._worker.stream_feed(self.session_id, samples)
        except Exception:
            # A failed feed leaves the engine in an unknown state, so the
            # session is abandoned rather than left claimable.
            self.abort()
            raise

    def close(self) -> StreamResult:
        if self._finished:
            raise ProviderError(f"session for {self.provider} is already closed")
        try:
            result = self._worker.stream_close(self.session_id)
            if not result.implementation:
                result.implementation = self.implementation
            return result
        finally:
            self._release()

    def abort(self) -> None:
        """Drop the session without finalizing it."""
        self._release()

    def _release(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._worker.release_stream()

    @property
    def finished(self) -> bool:
        return self._finished


class WorkerPool:
    """Lazily started, one worker per provider, shared by the whole process."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._workers: dict[str, ProviderWorker] = {}
        self._guard = threading.Lock()

    def worker(self, name: str) -> ProviderWorker:
        with self._guard:
            worker = self._workers.get(name)
            if worker is None:
                # Raises KeyError with a helpful message for unknown providers.
                self.settings.provider(name)
                worker = ProviderWorker(name, self.settings)
                self._workers[name] = worker
            return worker

    def run(
        self,
        name: str,
        audio: PreparedAudio,
        options: TranscriptionOptions,
    ) -> TranscriptionResult:
        return self.worker(name).transcribe(audio, options)

    def open_stream(self, name: str, options: StreamOptions) -> ProviderStream:
        """Open a live session for one provider, or fail fast if it is busy."""
        return self.worker(name).open_stream(options)

    def warm(self, names: list[str]) -> dict[str, str]:
        """Load the named providers now, reporting per-provider failures.

        Used by the explicit warm-up command and before a benchmark, so model
        loading is never counted as inference latency.
        """
        outcomes: dict[str, str] = {}
        for name in names:
            try:
                self.worker(name).start()
                outcomes[name] = "ready"
            except (ProviderError, ProviderUnavailable, ProviderTimeout) as exc:
                outcomes[name] = str(exc)
        return outcomes

    def status(self) -> list[dict]:
        with self._guard:
            return [worker.status.as_dict() for worker in self._workers.values()]

    def shutdown(self) -> None:
        with self._guard:
            workers = list(self._workers.values())
        for worker in workers:
            worker.stop()


_pool: WorkerPool | None = None
_pool_guard = threading.Lock()


def get_pool(settings: Settings) -> WorkerPool:
    """Process-wide pool, so a model is loaded once and reused."""
    global _pool
    with _pool_guard:
        if _pool is None or _pool.settings.root != settings.root:
            if _pool is not None:
                _pool.shutdown()
            _pool = WorkerPool(settings)
            atexit.register(_pool.shutdown)
        return _pool
