"""Benchmark run endpoints."""

from __future__ import annotations

import csv
import io
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from app.api.deps import get_config, get_repository, get_worker_pool
from app.config import Settings
from app.domain.models import TranscriptionOptions
from app.scoring.normalize import NormalizationPolicy
from app.services.benchmark import RunRequest, execute_run, summarize_score
from app.storage.repository import Repository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["runs"])


def _split_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]


def store_upload(upload: UploadFile, settings: Settings) -> Path:
    """Save an upload under the data directory with a collision-free name."""
    uploads = settings.audio_dir / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    safe_name = Path(upload.filename or "audio.wav").name
    target = uploads / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    with target.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return target


def build_request(
    settings: Settings,
    *,
    audio_path: Path,
    audio_name: str | None,
    providers: str | None,
    word_timestamps: bool,
    language: str | None,
    keywords: str | None,
    reference_text: str | None,
    notes: str | None,
    normalization: str | None = None,
) -> RunRequest:
    names = _split_list(providers)
    if not names:
        raise HTTPException(status_code=400, detail="select at least one provider")

    unknown = [name for name in names if name not in settings.providers]
    if unknown:
        known = ", ".join(sorted(settings.providers)) or "none"
        raise HTTPException(
            status_code=400,
            detail=f"unknown provider(s): {', '.join(unknown)} (known: {known})",
        )

    try:
        policy = NormalizationPolicy(normalization) if normalization else None
    except ValueError:
        allowed = ", ".join(item.value for item in NormalizationPolicy)
        raise HTTPException(
            status_code=400,
            detail=f"unknown normalization policy {normalization!r} (allowed: {allowed})",
        ) from None

    request = RunRequest(
        audio_path=audio_path,
        audio_name=audio_name,
        providers=names,
        options=TranscriptionOptions(
            language=language or None,
            word_timestamps=word_timestamps,
            keywords=tuple(_split_list(keywords)),
        ),
        reference_text=reference_text or None,
        notes=notes or None,
    )
    if policy is not None:
        request.score_policy = policy
    return request


@router.post("/runs")
def create_run(
    file: UploadFile | None = File(default=None),
    audio_path: str | None = Form(default=None),
    providers: str = Form(default=""),
    word_timestamps: bool = Form(default=True),
    language: str | None = Form(default=None),
    keywords: str | None = Form(default=None),
    reference_text: str | None = Form(default=None),
    notes: str | None = Form(default=None),
    normalization: str | None = Form(default=None),
    settings: Settings = Depends(get_config),
    pool=Depends(get_worker_pool),
    repository: Repository = Depends(get_repository),
) -> JSONResponse:
    """Run one audio file through the selected providers.

    Synchronous by design for this milestone: a model load plus a decode is a
    long request, so the endpoint is a plain ``def`` and FastAPI runs it on a
    worker thread rather than blocking the event loop.
    """
    if file is not None and file.filename:
        target = store_upload(file, settings)
        name = Path(file.filename).name
    elif audio_path:
        target = Path(audio_path)
        if not target.is_absolute():
            target = settings.root / target
        if not target.exists():
            raise HTTPException(status_code=400, detail=f"audio file not found: {target}")
        name = target.name
    else:
        raise HTTPException(status_code=400, detail="supply a file or an audio_path")

    request = build_request(
        settings,
        audio_path=target,
        audio_name=name,
        providers=providers,
        word_timestamps=word_timestamps,
        language=language,
        keywords=keywords,
        reference_text=reference_text,
        notes=notes,
        normalization=normalization,
    )
    run = execute_run(settings, pool, repository, request)
    run["summary"] = summarize_score(run["results"])
    return JSONResponse(run)


@router.get("/runs")
def list_runs(
    limit: int = 50,
    repository: Repository = Depends(get_repository),
) -> dict:
    limit = max(1, min(limit, 500))
    return {"runs": repository.list_runs(limit)}


@router.get("/runs/{run_id}")
def get_run(
    run_id: str,
    repository: Repository = Depends(get_repository),
) -> dict:
    run = repository.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no run with id {run_id}")
    run["summary"] = summarize_score(run["results"])
    return run


@router.delete("/runs/{run_id}")
def delete_run(
    run_id: str,
    repository: Repository = Depends(get_repository),
) -> dict:
    if not repository.delete_run(run_id):
        raise HTTPException(status_code=404, detail=f"no run with id {run_id}")
    return {"deleted": run_id}


@router.get("/runs/{run_id}/export", response_model=None)
def export_run(
    run_id: str,
    format: str = "json",
    repository: Repository = Depends(get_repository),
) -> JSONResponse | PlainTextResponse:
    run = repository.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no run with id {run_id}")
    run["summary"] = summarize_score(run["results"])

    if format == "json":
        return JSONResponse(run)
    if format != "csv":
        raise HTTPException(status_code=400, detail="format must be json or csv")

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "run_id",
            "created_at",
            "provider",
            "model",
            "model_version",
            "transport",
            "language",
            "audio_duration_ms",
            "latency_ms",
            "time_to_first_token_ms",
            "real_time_factor",
            "wer",
            "cer",
            "reference_words",
            "substitutions",
            "insertions",
            "deletions",
            "truncated",
            "error",
            "text",
        ]
    )
    for result in run["results"]:
        score = result.get("score") or {}
        writer.writerow(
            [
                run["id"],
                run["created_at"],
                result.get("provider"),
                result.get("model"),
                result.get("model_version"),
                result.get("transport"),
                result.get("language"),
                result.get("audio_duration_ms"),
                result.get("latency_ms"),
                result.get("time_to_first_token_ms"),
                result.get("real_time_factor"),
                score.get("wer"),
                score.get("cer"),
                score.get("reference_words"),
                score.get("substitutions"),
                score.get("insertions"),
                score.get("deletions"),
                result.get("truncated"),
                result.get("error"),
                result.get("text"),
            ]
        )
    return PlainTextResponse(
        buffer.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="run-{run_id}.csv"'
        },
    )
