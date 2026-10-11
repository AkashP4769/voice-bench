"""Server-rendered UI.

HTMX posts the form and swaps in a results fragment, so the page never reloads
and there is no build step or Node toolchain involved.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from app.api.deps import get_config, get_repository, get_worker_pool
from app.api.routes_runs import build_request, store_upload
from app.config import Settings
from app.errors import VoiceBenchError
from app.providers.registry import HUMAN_NAMES, describe_configuration
from app.scoring.normalize import DEFAULT_POLICY, NormalizationPolicy
from app.services.benchmark import execute_run, summarize_score
from app.storage.repository import Repository
from app.ui.render import templates

router = APIRouter(tags=["ui"])

LANGUAGE_CHOICES = [
    ("auto", "Detect automatically"),
    ("en", "English"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("it", "Italian"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
]

# What the models are, for a reader who is about to compare them.
MODEL_NOTES = {
    "whistle": "16.9 MB, CPU, 7 languages, word timestamps and keyword biasing",
    "phonon": "164 MB, English-only, word timestamps and up to 25 hotwords",
}


def _provider_cards(settings: Settings, pool) -> list[dict]:
    workers = {entry["provider"]: entry for entry in pool.status()}
    cards = []
    for name, config in settings.providers.items():
        capabilities = describe_configuration(config)["capabilities"]
        languages = capabilities["languages"]
        cards.append(
            {
                "name": name,
                "display_name": HUMAN_NAMES.get(config.type, config.type),
                "model": config.model,
                "transport": config.transport,
                "note": MODEL_NOTES.get(config.type, ""),
                "languages": "any" if not languages else ", ".join(languages),
                "language_selection": capabilities["supports_language_selection"],
                "word_timestamps": capabilities["word_timestamps"],
                "hotwords": capabilities["hotwords"],
                "max_hotwords": capabilities["max_hotwords"],
                "max_duration_s": capabilities["max_duration_s"],
                "worker": workers.get(name),
            }
        )
    return cards


@router.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    settings: Settings = Depends(get_config),
    pool=Depends(get_worker_pool),
    repository: Repository = Depends(get_repository),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "providers": _provider_cards(settings, pool),
            "languages": LANGUAGE_CHOICES,
            "policies": [
                {"value": item.value, "is_default": item is DEFAULT_POLICY}
                for item in NormalizationPolicy
            ],
            "defaults": settings.defaults,
            "runs": repository.list_runs(15),
        },
    )


@router.get("/realtime", response_class=HTMLResponse)
def realtime_page(
    request: Request,
    settings: Settings = Depends(get_config),
    pool=Depends(get_worker_pool),
) -> HTMLResponse:
    from app.providers.phonon import live_session_available

    streamable = []
    for name, config in settings.providers.items():
        capabilities = describe_configuration(config)["capabilities"]
        if not capabilities["streaming"]:
            continue
        streamable.append(
            {
                "name": name,
                "display_name": HUMAN_NAMES.get(config.type, config.type),
                "model": config.model,
                "languages": capabilities["languages"],
                "language_selection": capabilities["supports_language_selection"],
                "hotwords": capabilities["hotwords"],
                # Word timings are available live from Whistle only; Phonon's
                # live session reports text without them.
                "stream_word_timestamps": config.type == "whistle",
            }
        )

    phonon_note = None
    if any(entry["name"].startswith("phonon") for entry in streamable):
        phonon_note = (
            "Fermion's own live session is in use."
            if live_session_available()
            else "Fermion's live session is unavailable in this build, so Phonon "
            "uses this harness's rolling segmenter as a fallback."
        )

    return templates.TemplateResponse(
        request,
        "realtime.html",
        {
            "providers": streamable,
            "languages": LANGUAGE_CHOICES,
            "phonon_note": phonon_note,
            # Whistle measured at ~1.07x realtime on 1 s chunks and ~0.63x on 2 s,
            # so the default cadence is the one that keeps up.
            "cadences": [1.0, 2.0, 3.0],
            "default_cadence": 2.0,
        },
    )


@router.post("/ui/runs", response_class=HTMLResponse)
def ui_run(
    request: Request,
    file: UploadFile | None = File(default=None),
    audio_path: str | None = Form(default=None),
    providers: list[str] = Form(default=[]),
    word_timestamps: bool = Form(default=False),
    language: str | None = Form(default=None),
    keywords: str | None = Form(default=None),
    reference_text: str | None = Form(default=None),
    notes: str | None = Form(default=None),
    normalization: str | None = Form(default=None),
    settings: Settings = Depends(get_config),
    pool=Depends(get_worker_pool),
    repository: Repository = Depends(get_repository),
) -> HTMLResponse:
    def error(message: str, status_code: int = 400) -> HTMLResponse:
        # Rendered into the results slot so a validation problem appears inline
        # instead of replacing the page with a JSON error.
        return templates.TemplateResponse(
            request, "_error.html", {"message": message}, status_code=status_code
        )

    if not providers:
        return error("Select at least one provider.")

    if file is not None and file.filename:
        target = store_upload(file, settings)
        name = file.filename
    elif audio_path:
        from pathlib import Path

        target = Path(audio_path)
        if not target.is_absolute():
            target = settings.root / target
        if not target.exists():
            return error(f"Audio file not found: {target}")
        name = target.name
    else:
        return error("Choose an audio file.")

    try:
        run_request = build_request(
            settings,
            audio_path=target,
            audio_name=name,
            providers=",".join(providers),
            word_timestamps=word_timestamps,
            language=None if language in (None, "", "auto") else language,
            keywords=keywords,
            reference_text=reference_text,
            notes=notes,
            normalization=normalization,
        )
    except HTTPException as exc:
        return error(str(exc.detail))
    except VoiceBenchError as exc:
        return error(str(exc))

    try:
        run = execute_run(settings, pool, repository, run_request)
    except VoiceBenchError as exc:
        return error(str(exc), status_code=502)

    run["summary"] = summarize_score(run["results"])
    return templates.TemplateResponse(request, "_results.html", {"run": run})


@router.get("/runs/{run_id}", response_class=HTMLResponse)
def show_run(
    request: Request,
    run_id: str,
    repository: Repository = Depends(get_repository),
) -> HTMLResponse:
    run = repository.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no run with id {run_id}")
    run["summary"] = summarize_score(run["results"])
    return templates.TemplateResponse(
        request, "run.html", {"run": run}
    )


@router.get("/history", response_class=HTMLResponse)
def history(
    request: Request,
    repository: Repository = Depends(get_repository),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "_history.html", {"runs": repository.list_runs(30)}
    )


@router.post("/ui/runs/{run_id}/delete", response_class=HTMLResponse)
def delete_run(
    request: Request,
    run_id: str,
    repository: Repository = Depends(get_repository),
) -> HTMLResponse:
    repository.delete_run(run_id)
    return templates.TemplateResponse(
        request, "_history.html", {"runs": repository.list_runs(30)}
    )
