"""Template environment and display filters.

Formatting lives here so the templates stay declarative and a missing value is
always rendered as an em dash rather than as a misleading zero.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from app import __version__

logger = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = UI_DIR / "templates"
STATIC_DIR = UI_DIR / "static"
ICONS_DIR = UI_DIR / "icons"

DASH = "\u2014"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

_XML_DECLARATION = re.compile(r"<\?xml[^>]*\?>")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_PIXEL_SIZE = re.compile(r"\s(?:width|height)=\"[^\"]*\"")
_ICON_CACHE: dict[str, str] = {}


def icon(name: str) -> Markup:
    """Return an SVG from ``app/ui/icons`` as markup safe to inline.

    The source files carry a hardcoded ``fill="#000000"`` and pixel dimensions,
    which would render invisible on a dark theme and fight the CSS. Those are
    rewritten on load rather than in the asset, so the icons stay interchangeable
    with anything else that consumes plain SVG. The element is marked
    ``aria-hidden`` because the surrounding control carries the accessible name.

    The result is ``Markup`` so Jinja does not escape the SVG into visible text.
    The input is a file from this repository, not user data, so treating it as
    safe is appropriate.
    """
    if name in _ICON_CACHE:
        return _ICON_CACHE[name]

    empty = Markup("")
    path = ICONS_DIR / f"{name}.svg"
    if not path.exists():
        logger.warning("icon %r not found at %s", name, path)
        _ICON_CACHE[name] = empty
        return empty

    try:
        markup = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("icon %r could not be read: %s", name, exc)
        _ICON_CACHE[name] = empty
        return empty

    markup = _XML_DECLARATION.sub("", markup)
    markup = _COMMENT.sub("", markup)
    markup = _PIXEL_SIZE.sub("", markup)
    markup = markup.replace('fill="#000000"', 'fill="currentColor"')
    markup = markup.replace(
        "<svg", '<svg aria-hidden="true" focusable="false"', 1
    ).strip()

    rendered = Markup(markup)
    _ICON_CACHE[name] = rendered
    return rendered


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

templates.env.globals["icon"] = icon


def asset(path: str) -> str:
    """Return a static asset URL carrying the application version.

    Without a version string the browser happily reuses a cached stylesheet
    after an update, which makes a CSS change look like it had no effect at all.
    """
    return f"/static/{path.lstrip('/')}?v={__version__}"


templates.env.globals["asset"] = asset
templates.env.globals["asset_version"] = __version__
