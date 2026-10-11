"""The audio source controls and the recorder assets.

Recording itself happens in the browser and cannot be exercised here; what these
tests pin down is that both entry points exist, that the upload field the form
depends on survived the redesign, and that the icon is inlined in a form the
dark theme can actually show.
"""

from __future__ import annotations

import re

import pytest

from app.ui import render


def test_index_offers_both_audio_sources(client):
    body = client.get("/").text

    # The upload path: the input must still be named "file" for the backend,
    # and must carry an id the label can point at.
    assert re.search(r'id="audio-file"[^>]*name="file"', body) or re.search(
        r'name="file"[^>]*id="audio-file"', body
    ), "the file input lost its id or its name"
    assert 'for="audio-file"' in body
    assert "Choose audio file" in body

    # The microphone path.
    assert 'id="record-button"' in body
    assert 'aria-pressed="false"' in body
    assert "Record from microphone" in body


def test_recorder_controls_are_present(client):
    body = client.get("/").text
    for element in (
        'id="recorder"',
        'id="record-timer"',
        'id="record-level-fill"',
        'id="record-stop"',
        'id="record-clear"',
        'id="audio-status"',
        'id="run-button"',
    ):
        assert element in body, f"missing {element}"


def test_microphone_icon_is_inlined_and_theme_aware(client):
    body = client.get("/").text
    # Inlined as markup, not fetched as an <img>, so CSS can recolour it.
    assert "<svg" in body
    assert 'src="/icons/microphone.svg"' not in body
    # The asset is hardcoded black; it must be rewritten to inherit.
    assert 'fill="currentColor"' in body
    assert 'fill="#000000"' not in body
    # Sized by CSS instead of by the asset's pixel dimensions.
    assert not re.search(r"<svg[^>]*\swidth=", body)
    assert not re.search(r"<svg[^>]*\sheight=", body)
    # The button carries the accessible name, so the icon must be hidden.
    assert 'aria-hidden="true"' in body


def test_icon_keeps_the_drawing_intact():
    markup = render.icon("microphone")
    assert "viewBox" in markup
    assert "<path" in markup
    # Nothing stray from the source file leaks into the page.
    assert "<?xml" not in markup
    assert "<!--" not in markup


def test_icon_is_cached_across_renders():
    assert render.icon("microphone") is render.icon("microphone")


def test_missing_icon_is_empty_rather_than_an_error():
    assert render.icon("definitely-not-an-icon") == ""


def test_recorder_assets_are_served(client):
    for path, expected in (
        ("/static/recorder.js", "encodeWav"),
        ("/static/recorder-worklet.js", "registerProcessor"),
    ):
        response = client.get(path)
        assert response.status_code == 200, path
        assert expected in response.text, path


def test_page_loads_the_recorder_script(client):
    body = client.get("/").text
    assert "/static/recorder.js" in body
    # Deferred, so it runs after the markup it wires up exists.
    assert re.search(r'<script[^>]*recorder\.js[^>]*defer', body)


def test_static_assets_are_versioned(client):
    """A cached stylesheet makes a CSS change look like it had no effect."""
    body = client.get("/").text
    assert re.search(r'href="/static/app\.css\?v=[^"]+"', body)
    assert re.search(r'src="/static/recorder\.js\?v=[^"]+"', body)
    assert re.search(r'src="/static/htmx\.min\.js\?v=[^"]+"', body)


def test_recorder_script_receives_the_version_for_the_worklet(client):
    """The worklet is fetched by URL from JS, so it is versioned by hand."""
    body = client.get("/").text
    match = re.search(r'data-version="([^"]+)"', body)
    assert match, "the recorder script is missing its version attribute"

    source = (render.STATIC_DIR / "recorder.js").read_text(encoding="utf-8")
    assert "dataset.version" in source
    assert '"/static/recorder-worklet.js" + (SCRIPT_VERSION' in source


def test_hidden_elements_are_forced_hidden():
    """`.recorder { display: flex }` outranks the UA rule for `[hidden]`.

    Without this, the Stop/Discard panel is visible before recording starts.
    """
    css = (render.STATIC_DIR / "app.css").read_text(encoding="utf-8")
    assert "[hidden] { display: none !important; }" in css


def test_base_input_rule_excludes_the_visually_hidden_input():
    """`input[type="file"]` outranks `.visually-hidden` on width.

    Without the :not(), the hidden file input becomes a 100%-wide box that
    overflows the page and misaligns the source row.
    """
    css = (render.STATIC_DIR / "app.css").read_text(encoding="utf-8")
    assert 'input[type="file"]:not(.visually-hidden)' in css


def test_button_resets_the_inherited_label_margin():
    """Otherwise the label sits higher than its siblings under align-items."""
    css = (render.STATIC_DIR / "app.css").read_text(encoding="utf-8")
    block = css.split(".btn {", 1)[1].split("}", 1)[0]
    assert "margin-bottom: 0" in block


def test_recorder_does_not_mute_the_audio_it_captures():
    """Regression: a zero gain in the path produced recordings of pure silence.

    The worklet already outputs nothing, so a mute node is both unnecessary and
    destructive.
    """
    source = (render.STATIC_DIR / "recorder.js").read_text(encoding="utf-8")
    assert "createGain" not in source
    assert "source.connect(node)" in source


def test_recorder_warns_about_a_too_short_recording():
    source = (render.STATIC_DIR / "recorder.js").read_text(encoding="utf-8")
    assert "MIN_USEFUL_SECONDS" in source
    assert "very short" in source


def test_worklet_processor_name_matches_the_page_side():
    """A mismatch here would fail only at runtime, in the browser."""
    page = (render.STATIC_DIR / "recorder.js").read_text(encoding="utf-8")
    worklet = (render.STATIC_DIR / "recorder-worklet.js").read_text(encoding="utf-8")

    assert 'const PROCESSOR_NAME = "voicebench-recorder"' in page
    assert 'registerProcessor("voicebench-recorder"' in worklet


def test_recorder_targets_sixteen_kilohertz_mono():
    """Both engines expect 16 kHz mono, so this must not drift."""
    page = (render.STATIC_DIR / "recorder.js").read_text(encoding="utf-8")
    assert "const TARGET_SAMPLE_RATE = 16000;" in page
    # A 16-bit mono PCM WAV header, not a compressed container the backend
    # cannot decode.
    assert 'view.setUint16(22, 1, true); // channels: mono' in page
    assert "view.setUint16(20, 1, true); // format: PCM" in page


@pytest.mark.parametrize("asset", ["recorder.js", "recorder-worklet.js"])
def test_recorder_scripts_run_in_strict_mode(asset):
    """Strict mode turns an accidental global into a visible error."""
    source = (render.STATIC_DIR / asset).read_text(encoding="utf-8")
    assert '"use strict"' in source
