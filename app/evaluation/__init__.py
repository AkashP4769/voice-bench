"""Fetching a small ground-truthed set and evaluating providers against it."""

from app.evaluation.report import (
    ERROR,
    OK,
    UNSUPPORTED,
    ClipResult,
    ProviderSummary,
    render_language_table,
    render_table,
    summarise,
    summarise_all,
    to_csv,
    to_json,
)
from app.evaluation.run import EvaluationOptions, evaluate
from app.evaluation.sources import (
    CATALOG,
    CATALOG_BY_KEY,
    DEFAULT_SOURCES,
    Clip,
    FetchReport,
    Source,
    fetch_set,
    probe_source,
    read_manifest,
    write_manifest,
)

__all__ = [
    "CATALOG",
    "CATALOG_BY_KEY",
    "DEFAULT_SOURCES",
    "ERROR",
    "OK",
    "UNSUPPORTED",
    "Clip",
    "ClipResult",
    "EvaluationOptions",
    "FetchReport",
    "ProviderSummary",
    "Source",
    "evaluate",
    "fetch_set",
    "probe_source",
    "read_manifest",
    "render_language_table",
    "render_table",
    "summarise",
    "summarise_all",
    "to_csv",
    "to_json",
    "write_manifest",
]
