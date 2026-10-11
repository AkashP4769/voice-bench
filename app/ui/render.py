"""Template environment and display filters.

Formatting lives here so the templates stay declarative and a missing value is
always rendered as an em dash rather than as a misleading zero.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

DASH = "\u2014"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def fmt_ms(value: float | None, digits: int = 0) -> str:
    if value is None:
        return DASH
    if value >= 1000:
        return f"{value / 1000:.2f} s"
    return f"{value:.{digits}f} ms"


def fmt_sec(value: float | None, digits: int = 1) -> str:
    """Model load time is reported in seconds, not milliseconds."""
    if value is None:
        return DASH
    return f"{value:.{digits}f} s"


def fmt_num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return DASH
    return f"{value:.{digits}f}"


def fmt_ratio(value: float | None) -> str:
    if value is None:
        return DASH
    return f"{value * 100:.2f}%"


templates.env.filters.update(
    {
        "ms": fmt_ms,
        "sec": fmt_sec,
        "num": fmt_num,
        "ratio": fmt_ratio,
    }
)
