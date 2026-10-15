"""The audio source controls and the recorder assets.

Recording itself happens in the browser and cannot be exercised here; what these
tests pin down is that both entry points exist, that the upload field the form
depends on survived the redesign, and that the icon is inlined in a form the
dark theme can actually show.
"""

from __future__ import annotations

import json
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


def test_automated_benchmark_page_offers_librispeech(client):
    response = client.get("/benchmarks")
    assert response.status_code == 200
    body = response.text
    assert "Automated benchmark" in body
    assert 'name="benchmark_provider" value="librispeech"' in body
    assert 'value="librispeech-clean"' in body
    assert 'value="librispeech-other"' in body
    assert 'hx-post="/ui/benchmarks"' in body


def test_automated_benchmark_post_renders_results(client, settings, monkeypatch):
    from app.api import routes_ui
    from app.evaluation.report import ClipResult
    from app.evaluation.sources import Clip, FetchReport

    clip = Clip(
        clip_id="librispeech-clean-00",
        source="librispeech-clean",
        language="en",
        reference="one two",
        path=str(settings.data_dir / "clip.flac"),
    )
    row = ClipResult(
        clip_id=clip.clip_id,
        source=clip.source,
        language=clip.language,
        provider="whistle",
        reference=clip.reference,
        text=clip.reference,
        wer=0.0,
        cer=0.0,
        reference_words=2,
    )
    monkeypatch.setattr(
        routes_ui,
        "fetch_set",
        lambda *args, **kwargs: FetchReport(clips=[clip]),
    )
    monkeypatch.setattr(
        routes_ui,
        "evaluate",
        lambda *args, **kwargs: [row],
    )

    response = client.post(
        "/ui/benchmarks",
        data={
            "benchmark_provider": "librispeech",
            "source_keys": "librispeech-clean",
            "per_source": "1",
            "providers": "whistle",
        },
    )
    assert response.status_code == 200
    assert "LibriSpeech results" in response.text
    assert "librispeech-clean-00" in response.text
    assert "0.00%" in response.text


def run_benchmark(client, monkeypatch, **overrides):
    """Post one automated run with the network and the models stubbed out."""
    from app.api import routes_ui
    from app.evaluation.report import ClipResult
    from app.evaluation.sources import Clip, FetchReport

    clip = Clip(
        clip_id="librispeech-clean-00",
        source="librispeech-clean",
        language="en",
        reference="one two",
        path="clip.flac",
    )
    row = ClipResult(
        clip_id=clip.clip_id,
        source=clip.source,
        language=clip.language,
        provider="whistle",
        reference=clip.reference,
        text=clip.reference,
        wer=0.0,
        cer=0.0,
        reference_words=2,
    )
    monkeypatch.setattr(
        routes_ui, "fetch_set", lambda *args, **kwargs: FetchReport(clips=[clip])
    )
    monkeypatch.setattr(routes_ui, "evaluate", lambda *args, **kwargs: [row])

    data = {
        "benchmark_provider": "librispeech",
        "source_keys": "librispeech-clean",
        "per_source": "1",
        "providers": "whistle",
    }
    data.update(overrides)
    return client.post("/ui/benchmarks", data=data)


def test_a_web_benchmark_run_is_written_to_disk(client, settings, monkeypatch):
    """The result must outlive the request that produced it."""
    response = run_benchmark(client, monkeypatch)
    assert response.status_code == 200

    directory = settings.data_dir / "benchmarks" / "librispeech"
    runs = sorted((directory / "runs").glob("*.json"))
    assert len(runs) == 1
    assert runs[0].with_suffix(".csv").exists()
    assert (directory / "latest.json").exists()
    assert (directory / "latest.csv").exists()

    stored = json.loads(runs[0].read_text(encoding="utf-8"))
    assert stored["metadata"]["benchmark"] == "librispeech"
    assert stored["metadata"]["providers"] == ["whistle"]
    assert stored["metadata"]["normalization"] == "lower_strip_punctuation"
    assert stored["metadata"]["run_id"] == runs[0].stem
    assert stored["summary"][0]["provider"] == "whistle"
    # The rendered page names the file it was written to.
    assert runs[0].stem in response.text


def test_each_web_run_keeps_its_own_record(client, settings, monkeypatch):
    run_benchmark(client, monkeypatch)
    run_benchmark(client, monkeypatch)

    directory = settings.data_dir / "benchmarks" / "librispeech" / "runs"
    assert len(sorted(directory.glob("*.json"))) == 2


def test_saved_runs_are_listed_on_the_benchmarks_page(client, monkeypatch):
    run_benchmark(client, monkeypatch)

    body = client.get("/benchmarks/runs").text
    assert "LibriSpeech" in body
    assert "whistle" in body
    assert "0.00%" in body
    assert "No saved runs yet" not in body


def test_the_listing_is_empty_before_any_run(client):
    assert "No saved runs yet" in client.get("/benchmarks/runs").text


def saved_run_ids(settings):
    """Stored run ids for the LibriSpeech provider, newest first."""
    directory = settings.data_dir / "benchmarks" / "librispeech" / "runs"
    return sorted((path.stem for path in directory.glob("*.json")), reverse=True)


def test_a_stored_run_can_be_viewed_after_a_restart(client, settings, monkeypatch):
    run_benchmark(client, monkeypatch)
    run_id = saved_run_ids(settings)[0]

    page = client.get(f"/benchmarks/runs/librispeech/{run_id}")
    assert page.status_code == 200
    assert "LibriSpeech results" in page.text
    assert "librispeech-clean-00" in page.text
    assert run_id in page.text
    # Read back from disk, not from the request that produced it.
    assert "Stored as" in page.text


def test_a_stored_run_exports_json_and_csv(client, settings, monkeypatch):
    run_benchmark(client, monkeypatch)
    run_id = saved_run_ids(settings)[0]

    as_json = client.get(f"/benchmarks/runs/librispeech/{run_id}/export?format=json")
    assert as_json.status_code == 200
    assert as_json.json()["metadata"]["run_id"] == run_id

    as_csv = client.get(f"/benchmarks/runs/librispeech/{run_id}/export?format=csv")
    assert as_csv.status_code == 200
    assert as_csv.text.splitlines()[0].startswith("clip_id,source,language,provider")
    assert "librispeech-clean-00" in as_csv.text


def test_export_rejects_an_unknown_format(client, settings, monkeypatch):
    run_benchmark(client, monkeypatch)
    run_id = saved_run_ids(settings)[0]
    response = client.get(f"/benchmarks/runs/librispeech/{run_id}/export?format=xls")
    assert response.status_code == 400


def test_an_unknown_stored_run_is_a_404(client):
    assert client.get("/benchmarks/runs/librispeech/nope").status_code == 404
    assert client.get("/benchmarks/nothing/20261015-000000").status_code == 404


def test_a_traversing_run_id_is_refused(client):
    """The id becomes a path segment, so it must not reach the filesystem."""
    response = client.get("/benchmarks/runs/librispeech/..%2Flatest")
    assert response.status_code == 404


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
