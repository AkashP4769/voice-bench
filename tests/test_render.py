"""Display formatting.

These exist because a unit mistake here is invisible in a screenshot: a model
load time of 3.3 seconds rendered as "3 ms" looks plausible and is wrong.
"""

from __future__ import annotations

from app.ui.render import fmt_ms, fmt_num, fmt_ratio, fmt_sec


def test_milliseconds_switch_to_seconds_above_a_thousand():
    assert fmt_ms(320.0) == "320 ms"
    assert fmt_ms(3375.0) == "3.38 s"


def test_milliseconds_missing_value_is_a_dash_not_zero():
    assert fmt_ms(None) == "\u2014"


def test_seconds_are_not_treated_as_milliseconds():
    """Regression: load_seconds went through the ms filter and showed "3 ms"."""
    assert fmt_sec(3.33) == "3.3 s"
    assert fmt_sec(0.86) == "0.9 s"
    assert fmt_sec(None) == "\u2014"


def test_ratios_render_as_percentages():
    assert fmt_ratio(0.0425) == "4.25%"
    assert fmt_ratio(0.0) == "0.00%"
    assert fmt_ratio(None) == "\u2014"


def test_numbers_respect_digits_and_missing_values():
    assert fmt_num(1.23456, 3) == "1.235"
    assert fmt_num(None) == "\u2014"
