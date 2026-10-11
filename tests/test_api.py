"""HTTP API and UI.

The worker pool is replaced with a stub here, so these tests exercise upload,
normalization, capability checks, scoring, persistence and rendering without
loading a model.
"""

from __future__ import annotations

import json

from app.domain.models import (
    TranscriptionResult,
    WindowingInfo,
    WindowingStrategy,
    Word,
)


def canned(provider: str, text: str, words: bool = True) -> TranscriptionResult:
    return TranscriptionResult(
        provider=provider,
        model=provider,
        model_version="test-1.0",
        transport="in_process",
        text=text,
        language="en",
        words=(
            [
                Word(text="the", start_ms=0.0, end_ms=120.0, probability=0.93),
                Word(text="quick", start_ms=120.0, end_ms=400.0, probability=0.88),
            ]
            if words
            else []
        ),
        audio_duration_ms=1000.0,
        latency_ms=123.4,
        time_to_first_token_ms=10.0,
        real_time_factor=0.1234,
        windowing=WindowingInfo(WindowingStrategy.SINGLE_PASS, window_count=1),
    )


def upload(path):
    return {"file": ("clip.wav", path.read_bytes(), "audio/wav")}


# -- health and discovery --------------------------------------------------


def test_health_reports_ok(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_providers_lists_both_models_with_capabilities(client):
    payload = client.get("/api/providers").json()
    names = {entry["name"] for entry in payload["providers"]}
    assert {"whistle", "phonon-2"} <= names

    phonon = next(e for e in payload["providers"] if e["name"] == "phonon-2")
    assert phonon["capabilities"]["languages"] == ["en"]
    assert phonon["capabilities"]["supports_language_selection"] is False

    whistle = next(e for e in payload["providers"] if e["name"] == "whistle")
    assert len(whistle["capabilities"]["languages"]) == 7


def test_providers_endpoint_never_returns_a_key_value(client, monkeypatch):
    secret = "sk-live-should-never-appear"
    monkeypatch.setenv("PHONON_API_KEY", secret)
    monkeypatch.setenv("WHISTLE_API_KEY", secret)
    body = client.get("/api/providers").text
    assert secret not in body


# -- validation ------------------------------------------------------------


def test_run_requires_audio_or_a_path(client):
    response = client.post("/api/runs", data={"providers": "whistle"})
    assert response.status_code == 400
    assert "file" in response.json()["detail"]


def test_run_requires_at_least_one_provider(client, synth_wav):
    response = client.post("/api/runs", files=upload(synth_wav), data={"providers": ""})
    assert response.status_code == 400
    assert "provider" in response.json()["detail"]


def test_run_rejects_unknown_provider(client, synth_wav):
    response = client.post(
        "/api/runs", files=upload(synth_wav), data={"providers": "whistle,ghost"}
    )
    assert response.status_code == 400
    assert "ghost" in response.json()["detail"]


def test_run_rejects_unknown_normalization_policy(client, synth_wav):
    response = client.post(
        "/api/runs",
        files=upload(synth_wav),
        data={"providers": "whistle", "normalization": "nonsense"},
    )
    assert response.status_code == 400
    assert "normalization" in response.json()["detail"]


def test_run_from_a_path_that_does_not_exist(client):
    response = client.post(
        "/api/runs", data={"providers": "whistle", "audio_path": "nope.wav"}
    )
    assert response.status_code == 400


# -- execution -------------------------------------------------------------


def test_run_returns_normalized_results_and_persists(client, synth_wav, repository):
    client.fake_pool.results = {
        "whistle": canned("whistle", "the quick brown fox"),
        "phonon-2": canned("phonon-2", "the quick brown fox"),
    }
    response = client.post(
        "/api/runs",
        files=upload(synth_wav),
        data={"providers": "whistle,phonon-2", "word_timestamps": "true"},
    )
    assert response.status_code == 200
    run = response.json()

    assert run["id"]
    assert len(run["results"]) == 2
    assert {r["provider"] for r in run["results"]} == {"whistle", "phonon-2"}
    for result in run["results"]:
        assert result["error"] is None
        assert result["transport"] == "in_process"
        assert result["audio_duration_ms"] == 1000.0
        assert result["words"]

    # The same run is retrievable afterwards.
    stored = client.get(f"/api/runs/{run['id']}").json()
    assert stored["id"] == run["id"]
    assert len(stored["results"]) == 2
    assert stored["audio"]["sha256"] == run["audio"]["sha256"]


def test_run_scores_against_a_reference_transcript(client, synth_wav):
    client.fake_pool.results = {
        "whistle": canned("whistle", "the quick brown fox"),
        "phonon-2": canned("phonon-2", "the quick brown box"),
    }
    run = client.post(
        "/api/runs",
        files=upload(synth_wav),
        data={
            "providers": "whistle,phonon-2",
            "reference_text": "the quick brown fox",
        },
    ).json()

    scores = {r["provider"]: r["score"] for r in run["results"]}
    assert scores["whistle"]["wer"] == 0.0
    assert scores["phonon-2"]["wer"] == 0.25
    assert run["summary"]["best_wer_provider"] == "whistle"


def test_run_without_a_reference_reports_no_word_error_rate(client, synth_wav):
    client.fake_pool.results = {"whistle": canned("whistle", "anything")}
    run = client.post(
        "/api/runs", files=upload(synth_wav), data={"providers": "whistle"}
    ).json()
    assert run["results"][0]["score"] is None
    assert any("no reference transcript" in caveat for caveat in run["caveats"])


def test_provider_failure_is_reported_per_provider_not_as_a_whole_run(
    client, synth_wav
):
    from app.errors import ProviderUnavailable

    client.fake_pool.results = {
        "whistle": canned("whistle", "the quick brown fox"),
        "phonon-2": ProviderUnavailable("fermion-research is not installed"),
    }
    response = client.post(
        "/api/runs",
        files=upload(synth_wav),
        data={"providers": "whistle,phonon-2"},
    )
    assert response.status_code == 200
    results = {r["provider"]: r for r in response.json()["results"]}
    assert results["whistle"]["error"] is None
    assert "not installed" in results["phonon-2"]["error"]


def test_capability_violation_is_reported_without_running_the_provider(
    client, synth_wav
):
    client.fake_pool.results = {"phonon-2": canned("phonon-2", "text")}
    run = client.post(
        "/api/runs",
        files=upload(synth_wav),
        data={"providers": "phonon-2", "language": "de"},
    ).json()
    result = run["results"][0]
    assert result["error"] is not None
    assert "language" in result["error"]
    # The provider was never invoked, so no model load was wasted.
    assert client.fake_pool.calls == []


def test_mixed_transports_are_flagged_as_not_comparable(client, synth_wav):
    first = canned("whistle", "a")
    second = canned("phonon-2", "b")
    second.transport = "openai_http"
    client.fake_pool.results = {"whistle": first, "phonon-2": second}
    run = client.post(
        "/api/runs", files=upload(synth_wav), data={"providers": "whistle,phonon-2"}
    ).json()
    assert any("different transports" in caveat for caveat in run["caveats"])


# -- listing and export ----------------------------------------------------


def test_list_runs_is_empty_initially(client):
    assert client.get("/api/runs").json()["runs"] == []


def test_export_json_and_csv(client, synth_wav):
    client.fake_pool.results = {"whistle": canned("whistle", "the quick brown fox")}
    run = client.post(
        "/api/runs",
        files=upload(synth_wav),
        data={"providers": "whistle", "reference_text": "the quick brown fox"},
    ).json()

    as_json = client.get(f"/api/runs/{run['id']}/export?format=json")
    assert as_json.status_code == 200
    assert as_json.json()["results"][0]["score"]["wer"] == 0.0

    as_csv = client.get(f"/api/runs/{run['id']}/export?format=csv")
    assert as_csv.status_code == 200
    assert "text/csv" in as_csv.headers["content-type"]
    lines = as_csv.text.strip().splitlines()
    assert lines[0].startswith("run_id,created_at,provider")
    assert "whistle" in lines[1]


def test_export_rejects_unknown_format(client, synth_wav):
    client.fake_pool.results = {"whistle": canned("whistle", "x")}
    run = client.post(
        "/api/runs", files=upload(synth_wav), data={"providers": "whistle"}
    ).json()
    assert client.get(f"/api/runs/{run['id']}/export?format=xml").status_code == 400


def test_unknown_run_is_a_404(client):
    assert client.get("/api/runs/does-not-exist").status_code == 404


def test_delete_removes_the_run(client, synth_wav):
    client.fake_pool.results = {"whistle": canned("whistle", "x")}
    run = client.post(
        "/api/runs", files=upload(synth_wav), data={"providers": "whistle"}
    ).json()
    assert client.delete(f"/api/runs/{run['id']}").status_code == 200
    assert client.get(f"/api/runs/{run['id']}").status_code == 404


def test_persisted_environment_records_package_versions(client, synth_wav):
    client.fake_pool.results = {"whistle": canned("whistle", "x")}
    run = client.post(
        "/api/runs", files=upload(synth_wav), data={"providers": "whistle"}
    ).json()
    assert "cactus-needle" in run["environment"]["packages"]
    assert run["environment"]["python"]


# -- UI --------------------------------------------------------------------


def test_index_renders_the_form_and_providers(client):
    body = client.get("/").text
    assert "Run a comparison" in body
    assert "Whistle" in body
    assert "Phonon-2" in body
    assert 'hx-post="/ui/runs"' in body


def test_index_preselects_the_default_normalization_policy(client):
    """A select always submits something, so the default must be marked."""
    import re

    body = client.get("/").text
    selected = re.findall(r'value="([^"]+)"\s*selected', body)
    assert "lower_strip_punctuation" in selected
    assert "none" not in selected


def test_run_without_a_normalization_choice_uses_the_documented_default(
    client, synth_wav
):
    """Case and punctuation must not decide a comparison unless asked to."""
    client.fake_pool.results = {"whistle": canned("whistle", "The Quick, Brown Fox!")}
    run = client.post(
        "/api/runs",
        files=upload(synth_wav),
        data={"providers": "whistle", "reference_text": "the quick brown fox"},
    ).json()

    result = run["results"][0]
    assert result["score"]["policy"] == "lower_strip_punctuation"
    assert result["score"]["wer"] == 0.0


def test_ui_run_renders_results_fragment(client, synth_wav):
    client.fake_pool.results = {"whistle": canned("whistle", "the quick brown fox")}
    response = client.post(
        "/ui/runs",
        files=upload(synth_wav),
        data={"providers": "whistle", "word_timestamps": "true"},
    )
    assert response.status_code == 200
    body = response.text
    assert "the quick brown fox" in body
    assert "Real-time factor" in body
    # The permalink for the new run is offered.
    assert "/api/runs/" in body


def test_ui_run_without_a_file_is_an_inline_error_not_a_crash(client):
    response = client.post("/ui/runs", data={"providers": "whistle"})
    assert response.status_code == 400
    assert "Choose an audio file" in response.text


def test_ui_run_without_providers_is_an_inline_error(client, synth_wav):
    response = client.post("/ui/runs", files=upload(synth_wav), data={})
    assert response.status_code == 400
    assert "Select at least one provider" in response.text


def test_ui_results_escape_provider_text(client, synth_wav):
    """Transcripts are model output and must not be able to inject markup."""
    client.fake_pool.results = {
        "whistle": canned("whistle", "<script>alert('x')</script>")
    }
    body = client.post(
        "/ui/runs", files=upload(synth_wav), data={"providers": "whistle"}
    ).text
    assert "<script>alert" not in body
    assert "&lt;script&gt;" in body


def test_saved_run_permalink_renders(client, synth_wav):
    client.fake_pool.results = {"whistle": canned("whistle", "hello there")}
    run = client.post(
        "/api/runs", files=upload(synth_wav), data={"providers": "whistle"}
    ).json()
    page = client.get(f"/runs/{run['id']}")
    assert page.status_code == 200
    assert "hello there" in page.text


def test_history_partial_lists_runs(client, synth_wav):
    client.fake_pool.results = {"whistle": canned("whistle", "x")}
    assert client.post(
        "/api/runs", files=upload(synth_wav), data={"providers": "whistle"}
    ).status_code == 200
    body = client.get("/history").text
    assert "whistle" in body
    assert "view" in body
