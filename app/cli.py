"""Command line interface.

Useful for scripted and batch benchmarking, and for warming models before a
timed comparison so a download is never mistaken for inference.

    bench providers
    bench warmup
    bench run clip.wav --providers whistle,phonon-2 --reference ground-truth.txt
    bench serve
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.domain.models import TranscriptionOptions
from app.errors import VoiceBenchError
from app.providers.registry import describe_configuration
from app.scoring.normalize import DEFAULT_POLICY, NormalizationPolicy
from app.services.benchmark import RunRequest, execute_run, summarize_score
from app.storage.repository import Repository
from app.workers.pool import get_pool

DASH = "\u2014"


def _fmt(value: float | None, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return DASH
    return f"{value:.{digits}f}{suffix}"


def _split(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def cmd_providers(_args: argparse.Namespace) -> int:
    settings = get_settings()
    rows = [describe_configuration(config) for config in settings.providers.values()]
    print(f"{'name':<12} {'type':<10} {'transport':<13} {'languages':<24} words  max audio")
    for row in rows:
        capabilities = row["capabilities"]
        languages = capabilities["languages"]
        language_text = "any" if not languages else ",".join(languages)
        max_audio = (
            f"{capabilities['max_duration_s']:.0f} s"
            if capabilities["max_duration_s"]
            else "engine segments"
        )
        print(
            f"{row['name']:<12} {row['type']:<10} {row['transport']:<13} "
            f"{language_text:<24} {str(capabilities['word_timestamps']):<6} {max_audio}"
        )
    return 0


def cmd_warmup(args: argparse.Namespace) -> int:
    settings = get_settings()
    names = args.provider or list(settings.providers)
    unknown = [name for name in names if name not in settings.providers]
    if unknown:
        known = ", ".join(sorted(settings.providers)) or "none"
        print(f"unknown provider(s): {', '.join(unknown)} (known: {known})", file=sys.stderr)
        return 2

    pool = get_pool(settings)
    print(f"loading {', '.join(names)}. The first run downloads weights.")
    outcomes = pool.warm(names)
    failures = 0
    for name, outcome in outcomes.items():
        if outcome == "ready":
            worker = pool.worker(name)
            load_seconds = worker.status.load_seconds or 0.0
            print(f"  {name}: ready in {load_seconds:.1f}s")
        else:
            failures += 1
            print(f"  {name}: FAILED. {outcome}", file=sys.stderr)
    pool.shutdown()
    return 1 if failures else 0


def cmd_env(_args: argparse.Namespace) -> int:
    from app.environment import collect, phonon_cpu_tier

    report = collect()
    report["phonon_engine"] = phonon_cpu_tier()
    print(json.dumps(report, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    settings = get_settings()
    repository = Repository(settings.db_path)
    pool = get_pool(settings)

    names = _split(args.providers)
    if not names:
        print("--providers is required, for example --providers whistle,phonon-2", file=sys.stderr)
        return 2

    reference = None
    if args.reference:
        reference = Path(args.reference).read_text(encoding="utf-8").strip()
    elif args.reference_text:
        reference = args.reference_text

    request = RunRequest(
        audio_path=Path(args.audio),
        audio_name=Path(args.audio).name,
        providers=names,
        options=TranscriptionOptions(
            language=args.language,
            word_timestamps=not args.no_word_timestamps,
            keywords=tuple(_split(args.keywords)),
        ),
        reference_text=reference,
        notes=args.notes,
        score_policy=NormalizationPolicy(args.normalization),
    )

    # Load models first so the reported latency is decode time, not startup.
    outcomes = pool.warm(names)
    for name, outcome in outcomes.items():
        if outcome != "ready":
            print(f"warning: {name} could not be loaded: {outcome}", file=sys.stderr)

    run = execute_run(settings, pool, repository, request)
    run["summary"] = summarize_score(run["results"])

    print()
    print(f"run {run['id']}  {run['audio']['name']}  {run['audio']['duration_ms']:.0f} ms audio")
    print(f"audio sha256 {run['audio']['sha256'][:16]}")
    print()
    header = f"{'provider':<12} {'status':<8} {'transport':<12} {'latency':>10} {'ttft':>9} {'rtf':>7} {'wer':>7} {'cer':>7} words"
    print(header)
    print("-" * len(header))
    for result in run["results"]:
        score = result.get("score") or {}
        wer = f"{score['wer'] * 100:.2f}%" if score.get("wer") is not None else DASH
        cer = f"{score['cer'] * 100:.2f}%" if score.get("cer") is not None else DASH
        status = "failed" if result["error"] else "ok"
        print(
            f"{result['provider']:<12} {status:<8} "
            f"{str(result['transport'] or DASH):<12} "
            f"{_fmt(result['latency_ms'], 1, ' ms'):>10} "
            f"{_fmt(result['time_to_first_token_ms'], 1, ' ms'):>9} "
            f"{_fmt(result['real_time_factor'], 3):>7} "
            f"{wer:>7} {cer:>7} {len(result['words']):>5}"
        )

    for result in run["results"]:
        print()
        print(f"--- {result['provider']} ---")
        if result["error"]:
            print(f"error: {result['error']}")
            continue
        for warning in result["warnings"]:
            print(f"warning: {warning}")
        print(result["text"] or "(empty transcript)")

    if run["caveats"]:
        print()
        print("caveats:")
        for caveat in run["caveats"]:
            print(f"  - {caveat}")

    if run["summary"]:
        print()
        print(
            f"lowest word error rate: {run['summary']['best_wer_provider']} "
            f"at {run['summary']['best_wer'] * 100:.2f}%"
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(run, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    if args.csv_out:
        _write_csv(run, Path(args.csv_out))
        print(f"wrote {args.csv_out}")

    pool.shutdown()
    return 0


def _write_csv(run: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["provider", "model", "transport", "latency_ms", "ttft_ms", "rtf",
             "wer", "cer", "truncated", "error", "text"]
        )
        for result in run["results"]:
            score = result.get("score") or {}
            writer.writerow(
                [
                    result["provider"],
                    result["model"],
                    result["transport"],
                    result["latency_ms"],
                    result["time_to_first_token_ms"],
                    result["real_time_factor"],
                    score.get("wer"),
                    score.get("cer"),
                    result["truncated"],
                    result["error"],
                    result["text"],
                ]
            )


def cmd_list(args: argparse.Namespace) -> int:
    settings = get_settings()
    repository = Repository(settings.db_path)
    runs = repository.list_runs(args.limit)
    if not runs:
        print("no runs yet")
        return 0
    print(f"{'id':<14} {'created':<22} {'audio':<28} {'providers':<24} results")
    for run in runs:
        name = (run["audio_name"] or DASH)[:27]
        print(
            f"{run['id']:<14} {run['created_at'][:19]:<22} {name:<28} "
            f"{','.join(run['providers'])[:23]:<24} {run.get('result_count', 0)}"
        )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    settings = get_settings()
    repository = Repository(settings.db_path)
    run = repository.get_run(args.run_id)
    if run is None:
        print(f"no run with id {args.run_id}", file=sys.stderr)
        return 1
    run["summary"] = summarize_score(run["results"])
    print(json.dumps(run, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench",
        description="Benchmark Phonon-2 and Whistle locally.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("providers", help="list configured providers and capabilities").set_defaults(
        func=cmd_providers
    )

    warm = sub.add_parser("warmup", help="download and load models before benchmarking")
    warm.add_argument("--provider", action="append", help="repeatable; default is all")
    warm.set_defaults(func=cmd_warmup)

    sub.add_parser("env", help="print machine and package facts").set_defaults(func=cmd_env)

    run = sub.add_parser("run", help="benchmark one audio file")
    run.add_argument("audio", help="path to an audio file")
    run.add_argument("--providers", required=True, help="comma-separated provider names")
    run.add_argument("--reference", help="file containing the ground-truth transcript")
    run.add_argument("--reference-text", help="ground-truth transcript as a string")
    run.add_argument("--language", help="force a language where supported")
    run.add_argument("--keywords", help="comma-separated hotwords")
    run.add_argument("--no-word-timestamps", action="store_true")
    run.add_argument("--notes", help="label stored with the run")
    run.add_argument(
        "--normalization",
        default=DEFAULT_POLICY.value,
        choices=[item.value for item in NormalizationPolicy],
        help="scoring normalization policy",
    )
    run.add_argument("--json-out", help="write the full run as JSON")
    run.add_argument("--csv-out", help="write a summary CSV")
    run.set_defaults(func=cmd_run)

    listing = sub.add_parser("list", help="list stored runs")
    listing.add_argument("--limit", type=int, default=20)
    listing.set_defaults(func=cmd_list)

    show = sub.add_parser("show", help="print one stored run as JSON")
    show.add_argument("run_id")
    show.set_defaults(func=cmd_show)

    serve = sub.add_parser("serve", help="run the web UI and API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except VoiceBenchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
