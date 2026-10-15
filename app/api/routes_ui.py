"""Server-rendered UI.

HTMX posts the form and swaps in a results fragment, so the page never reloads
and there is no build step or Node toolchain involved.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from app.api.deps import get_config, get_repository, get_worker_pool
from app.api.routes_runs import build_request, store_upload
from app.config import Settings
from app.evaluation import (
    BENCHMARK_PROVIDERS,
    BENCHMARK_PROVIDERS_BY_KEY,
    CATALOG_BY_KEY,
    ClipResult,
    EvaluationOptions,
    build_benchmark_payload,
    describe_benchmark_run,
    evaluate,
    fetch_set,
    list_benchmark_runs,
    load_benchmark_run,
    save_benchmark_run,
    summarise_all,
    to_csv,
    write_manifest,
)
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


def _benchmark_cards() -> list[dict]:
    """Build dataset choices without coupling the page to one corpus."""
    cards = []
    for provider in BENCHMARK_PROVIDERS:
        sources = []
        for source_key in provider.source_keys:
            source = CATALOG_BY_KEY[source_key]
            sources.append(
                {
                    "key": source.key,
                    "label": f"{source.config.title()} test subset",
                    "language": source.language,
                    "note": source.note,
                }
            )
        cards.append(
            {
                "key": provider.key,
                "name": provider.name,
                "note": provider.note,
                "sources": sources,
            }
        )
    return cards


def _benchmark_context(payload: dict, stored=None) -> dict:
    """One view model for a run, whether it was just measured or read back."""
    metadata = payload.get("metadata") or {}
    summaries = payload.get("summary") or []
    languages = sorted(
        {
            language
            for summary in summaries
            for language in (summary.get("by_language") or {})
        }
    )
    return {
        "metadata": metadata,
        "summaries": summaries,
        "rows": payload.get("clips") or [],
        "languages": languages,
        "stored": stored,
    }


def _benchmark_output_dir(settings: Settings, benchmark_key: str) -> Path:
    return settings.data_dir / "benchmarks" / benchmark_key


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


@router.get("/benchmarks", response_class=HTMLResponse)
def benchmarks_page(
    request: Request,
    settings: Settings = Depends(get_config),
    pool=Depends(get_worker_pool),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "benchmarks.html",
        {
            "providers": _provider_cards(settings, pool),
            "benchmark_providers": _benchmark_cards(),
            "policies": [
                {"value": item.value, "is_default": item is DEFAULT_POLICY}
                for item in NormalizationPolicy
            ],
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


@router.post("/ui/benchmarks", response_class=HTMLResponse)
def ui_benchmark(
    request: Request,
    benchmark_provider: str = Form(default="librispeech"),
    source_keys: list[str] = Form(default=[]),
    per_source: int = Form(default=2),
    limit: int = Form(default=0),
    providers: list[str] = Form(default=[]),
    normalization: str | None = Form(default=None),
    settings: Settings = Depends(get_config),
    pool=Depends(get_worker_pool),
) -> HTMLResponse:
    def error(message: str, status_code: int = 400) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "_error.html",
            {"message": message},
            status_code=status_code,
        )

    benchmark = BENCHMARK_PROVIDERS_BY_KEY.get(benchmark_provider)
    if benchmark is None:
        known = ", ".join(provider.key for provider in BENCHMARK_PROVIDERS)
        return error(
            f"Unknown benchmark provider {benchmark_provider!r} "
            f"(known: {known or 'none'})."
        )

    selected_sources = source_keys or list(benchmark.source_keys)
    invalid_sources = [
        key for key in selected_sources if key not in benchmark.source_keys
    ]
    if invalid_sources:
        return error(
            f"Source(s) {', '.join(invalid_sources)} are not part of "
            f"{benchmark.name}."
        )

    if not providers:
        return error("Select at least one model provider.")
    unknown = [name for name in providers if name not in settings.providers]
    if unknown:
        return error(f"Unknown model provider(s): {', '.join(unknown)}.")

    if not 1 <= per_source <= 20:
        return error("Clips per subset must be between 1 and 20.")
    if limit < 0:
        return error("Clip limit cannot be negative.")

    try:
        policy = NormalizationPolicy(normalization or DEFAULT_POLICY.value)
    except ValueError:
        return error(f"Unknown normalization policy: {normalization}.")

    output_dir = _benchmark_output_dir(settings, benchmark.key)
    report = fetch_set(
        selected_sources,
        output_dir,
        per_source=per_source,
        progress=None,
    )
    if not report.clips:
        detail = " ".join(report.failures) or "No usable clips were fetched."
        return error(f"{benchmark.name} could not be prepared: {detail}", 502)

    clips = report.clips[:limit] if limit else report.clips
    manifest = write_manifest(report.clips, output_dir / "manifest.jsonl")

    warm = pool.warm(providers)
    warm_failures = {
        name: status for name, status in warm.items() if status != "ready"
    }
    rows = evaluate(
        settings,
        pool,
        clips,
        EvaluationOptions(providers=providers, policy=policy),
        progress=None,
    )
    summaries = summarise_all(rows, providers)

    payload = build_benchmark_payload(
        benchmark_key=benchmark.key,
        benchmark_name=benchmark.name,
        source_keys=selected_sources,
        providers=providers,
        normalization=policy.value,
        summaries=summaries,
        rows=rows,
        clips=clips,
        manifest=manifest,
        fetched_clips=len(report.clips),
        fetch_failures=report.failures,
        warm_failures=warm_failures,
    )
    # Written before rendering: a result the page shows must exist on disk, so a
    # closed tab cannot lose it.
    stored = save_benchmark_run(output_dir, payload)

    return templates.TemplateResponse(
        request,
        "benchmark_results.html",
        _benchmark_context(payload, stored),
    )


@router.get("/benchmarks/runs", response_class=HTMLResponse)
def benchmark_runs(
    request: Request,
    settings: Settings = Depends(get_config),
) -> HTMLResponse:
    """Saved runs across every benchmark provider, newest first."""
    entries = []
    for benchmark in BENCHMARK_PROVIDERS:
        directory = _benchmark_output_dir(settings, benchmark.key)
        for stored in list_benchmark_runs(directory, limit=20):
            entries.append((stored.created_at, stored))
    entries.sort(key=lambda item: item[0], reverse=True)

    return templates.TemplateResponse(
        request,
        "_benchmark_runs.html",
        {"runs": [stored for _, stored in entries[:20]]},
    )


@router.get("/benchmarks/runs/{benchmark_key}/{run_id}", response_class=HTMLResponse)
def benchmark_run(
    request: Request,
    benchmark_key: str,
    run_id: str,
    settings: Settings = Depends(get_config),
) -> HTMLResponse:
    benchmark = BENCHMARK_PROVIDERS_BY_KEY.get(benchmark_key)
    if benchmark is None:
        raise HTTPException(status_code=404, detail=f"no benchmark {benchmark_key!r}")

    payload = load_benchmark_run(_benchmark_output_dir(settings, benchmark_key), run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"no stored run {run_id!r}")

    stored = describe_benchmark_run(
        payload, _benchmark_output_dir(settings, benchmark_key) / "runs" / f"{run_id}.json"
    )
    return templates.TemplateResponse(
        request,
        "benchmark_run.html",
        _benchmark_context(payload, stored),
    )


@router.get("/benchmarks/runs/{benchmark_key}/{run_id}/export", response_model=None)
def benchmark_run_export(
    benchmark_key: str,
    run_id: str,
    format: str = "json",
    settings: Settings = Depends(get_config),
) -> JSONResponse | PlainTextResponse:
    """The stored run, exactly as written, in either format."""
    benchmark = BENCHMARK_PROVIDERS_BY_KEY.get(benchmark_key)
    if benchmark is None:
        raise HTTPException(status_code=404, detail=f"no benchmark {benchmark_key!r}")

    payload = load_benchmark_run(_benchmark_output_dir(settings, benchmark_key), run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"no stored run {run_id!r}")

    if format == "json":
        return JSONResponse(payload)
    if format != "csv":
        raise HTTPException(status_code=400, detail="format must be json or csv")

    rows = [ClipResult(**clip) for clip in payload.get("clips") or []]
    return PlainTextResponse(
        to_csv(rows),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{benchmark_key}-{run_id}.csv"'
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
