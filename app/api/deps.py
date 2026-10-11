"""Shared FastAPI dependencies.

The repository and the worker pool are process-wide singletons: a model is
expensive to load, so it must be loaded once and reused across requests.
"""

from __future__ import annotations

import threading

from app.config import Settings, get_settings
from app.storage.repository import Repository
from app.workers.pool import WorkerPool, get_pool

_repository: Repository | None = None
_repository_lock = threading.Lock()


def get_config() -> Settings:
    return get_settings()


def get_repository() -> Repository:
    global _repository
    with _repository_lock:
        if _repository is None:
            _repository = Repository(get_settings().db_path)
        return _repository


def get_worker_pool() -> WorkerPool:
    return get_pool(get_settings())
