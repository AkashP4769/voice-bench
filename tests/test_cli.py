"""Command line interface.

The parser and the non-model commands are exercised here because a broken
import in ``cli.py`` is invisible to the API tests: the module is only imported
when someone actually runs ``bench``.
"""

from __future__ import annotations

import json

import pytest

import app.cli as cli


@pytest.fixture
def patched(monkeypatch, settings):
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    return settings


def test_parser_exposes_every_command():
    parser = cli.build_parser()
    for command in ("providers", "warmup", "env", "run", "list", "show", "serve"):
        args = parser.parse_args(_minimal_args(command))
        assert args.func is not None


def _minimal_args(command: str) -> list[str]:
    if command == "run":
        return ["run", "clip.wav", "--providers", "whistle"]
    if command == "show":
        return ["show", "some-run-id"]
    return [command]


def test_run_defaults_to_the_documented_normalization_policy():
    args = cli.build_parser().parse_args(
        ["run", "clip.wav", "--providers", "whistle"]
    )
    assert args.normalization == "lower_strip_punctuation"
    assert args.no_word_timestamps is False


def test_providers_command_lists_both_models(patched, capsys):
    assert cli.main(["providers"]) == 0
    output = capsys.readouterr().out
    assert "whistle" in output
    assert "phonon-2" in output
    assert "engine segments" in output


def test_warmup_rejects_an_unknown_provider(patched, capsys):
    assert cli.main(["warmup", "--provider", "ghost"]) == 2
    assert "unknown provider" in capsys.readouterr().err


def test_list_reports_an_empty_history(patched, capsys):
    assert cli.main(["list"]) == 0
    assert "no runs yet" in capsys.readouterr().out


def test_show_reports_an_unknown_run(patched, capsys):
    assert cli.main(["show", "missing"]) == 1
    assert "no run with id missing" in capsys.readouterr().err


def test_run_requires_providers(patched, capsys):
    assert cli.main(["run", "clip.wav", "--providers", ""]) == 2
    assert "--providers is required" in capsys.readouterr().err


def test_env_reports_package_versions(patched, capsys, monkeypatch):
    # Skip the optional Fermion engine probe; it starts a process.
    monkeypatch.setattr("app.environment.phonon_cpu_tier", lambda *a, **k: None)
    assert cli.main(["env"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["python"]
    assert "cactus-needle" in report["packages"]


def test_run_command_records_a_run(patched, synth_wav, capsys, monkeypatch):
    """The full CLI path, with the model pool replaced so nothing loads."""
    from app.domain.models import TranscriptionResult

    class StubPool:
        def warm(self, names):
            return {name: "ready" for name in names}

        def run(self, name, audio, options):
            return TranscriptionResult(
                provider=name,
                model=name,
                transport="in_process",
                text="the quick brown fox",
                audio_duration_ms=audio.duration_ms,
                latency_ms=42.0,
                real_time_factor=0.012,
            )

        def shutdown(self):
            pass

    monkeypatch.setattr(cli, "get_pool", lambda settings: StubPool())

    exit_code = cli.main(
        [
            "run",
            str(synth_wav),
            "--providers",
            "whistle",
            "--reference-text",
            "the quick brown fox",
            "--notes",
            "cli smoke",
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "lowest word error rate: whistle" in output
    assert "the quick brown fox" in output

    # And the run is retrievable afterwards.
    assert cli.main(["list"]) == 0
    assert "whistle" in capsys.readouterr().out
