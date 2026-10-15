"""The evaluation set: source selection, aggregation and reporting.

Network access is not exercised here. What matters and is tested is the logic
that decides which clips are used, how error rates aggregate, and that a
capability gap is reported as "skipped" rather than as a total failure.
"""

from __future__ import annotations

import json

import pytest

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
from app.evaluation.run import _unsupported_reason
from app.evaluation.sources import (
    CATALOG,
    CATALOG_BY_KEY,
    DEFAULT_SOURCES,
    Clip,
    Source,
    audio_url,
    select_varied_rows,
    sniff_extension,
    text_of,
    write_manifest,
    read_manifest,
)
from app.domain.provider import ProviderCapabilities


def row(text: str, speaker: str | None = "1", url: str = "https://example.test/a.flac") -> dict:
    return {
        "text": text,
        "transcript": text,
        "speaker_id": speaker,
        "audio": [{"src": url, "type": "audio/flac"}],
    }


# -- source catalogue ------------------------------------------------------


def test_catalogue_covers_both_english_and_whistles_languages():
    languages = {source.language for source in CATALOG}
    assert "en" in languages
    # Every language Whistle declares, apart from English, has a source.
    for language in ("de", "fr", "es", "it", "nl", "pl"):
        assert language in languages, language


def test_default_set_references_only_known_sources():
    for key in DEFAULT_SOURCES:
        assert key in CATALOG_BY_KEY, key


def test_source_keys_are_unique():
    keys = [source.key for source in CATALOG]
    assert len(keys) == len(set(keys))


def test_every_source_declares_where_its_transcript_lives():
    for source in CATALOG:
        assert source.text_field in {"text", "transcript"}
        assert source.offsets, source.key


# -- row handling ----------------------------------------------------------


def test_audio_url_is_read_from_the_list_shape():
    """The datasets-server returns audio as [{src, type}], not as a dict."""
    assert audio_url({"audio": [{"src": "https://x.test/a.flac"}]}) == "https://x.test/a.flac"
    assert audio_url({"audio": {"src": "https://x.test/b.flac"}}) == "https://x.test/b.flac"


def test_audio_url_is_none_when_absent():
    assert audio_url({}) is None
    assert audio_url({"audio": []}) is None
    assert audio_url({"audio": [{"type": "audio/flac"}]}) is None


def test_text_of_uses_the_sources_field_and_strips():
    source = CATALOG_BY_KEY["mls-german"]
    assert text_of({"transcript": "  hallo  "}, source.text_field) == "hallo"
    assert text_of({"text": "wrong field"}, source.text_field) == ""


def test_rows_without_a_transcript_or_audio_are_skipped():
    source = CATALOG_BY_KEY["librispeech-clean"]
    rows = [
        row(""),
        {"text": "no audio", "audio": []},
        row("good one"),
    ]
    chosen = select_varied_rows(rows, source, want=2)
    assert [text_of(r, source.text_field) for r in chosen] == ["good one"]


def test_selection_prefers_distinct_speakers():
    """Nine sentences from one voice measure that voice, not the model."""
    source = CATALOG_BY_KEY["librispeech-clean"]
    rows = [row("a", "1"), row("b", "1"), row("c", "2"), row("d", "3")]

    chosen = select_varied_rows(rows, source, want=2, seen_speakers=set())
    assert [r["speaker_id"] for r in chosen] == ["1", "2"]


def test_selection_avoids_speakers_already_used_by_this_source():
    source = CATALOG_BY_KEY["librispeech-clean"]
    rows = [row("a", "1"), row("b", "2")]
    chosen = select_varied_rows(rows, source, want=1, seen_speakers={"1"})
    assert chosen[0]["speaker_id"] == "2"


def test_selection_falls_back_rather_than_returning_nothing():
    """One speaker on the page still yields the clips that were asked for."""
    source = CATALOG_BY_KEY["librispeech-clean"]
    rows = [row("a", "1"), row("b", "1"), row("c", "1")]
    chosen = select_varied_rows(rows, source, want=3)
    assert len(chosen) == 3


def test_extension_is_sniffed_from_magic_bytes():
    assert sniff_extension(b"OggS.......") == ".opus"
    assert sniff_extension(b"fLaC.......") == ".flac"
    assert sniff_extension(b"RIFF.......") == ".wav"
    assert sniff_extension(b"ID3........") == ".mp3"
    assert sniff_extension(b"????") == ".bin"


def test_manifest_round_trips_unicode():
    clips = [
        Clip(
            clip_id="mls-polish-00",
            source="mls-polish",
            language="pl",
            reference="staliśmy wszyscy bezradni",
            path="C:/tmp/a.opus",
            duration_s=18.2,
            speaker_id="1889",
        )
    ]
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.jsonl"
        write_manifest(clips, path)
        restored = read_manifest(path)
        raw = path.read_text(encoding="utf-8")

    assert restored == clips
    # Written as real UTF-8, not escaped, so the file stays readable.
    assert "staliśmy" in raw


# -- capability gating -----------------------------------------------------


def test_english_only_provider_skips_other_languages():
    reason = _unsupported_reason(ProviderCapabilities(languages=("en",)), "de")
    assert reason == "en only"


def test_multilingual_provider_reports_its_language_list():
    capabilities = ProviderCapabilities(
        languages=("en", "de", "fr", "es", "it", "nl", "pl")
    )
    assert _unsupported_reason(capabilities, "de") is None
    assert _unsupported_reason(capabilities, "ru") == (
        "supports en, de, fr, es, it, nl, pl"
    )


def test_provider_without_a_language_restriction_accepts_anything():
    assert _unsupported_reason(ProviderCapabilities(languages=None), "ru") is None


# -- aggregation -----------------------------------------------------------


def scored(provider: str, language: str, ref_words: int, errors: int, **kwargs) -> ClipResult:
    return ClipResult(
        clip_id=f"{language}-{provider}",
        source="test",
        language=language,
        provider=provider,
        status=OK,
        reference_words=ref_words,
        word_errors=errors,
        wer=errors / ref_words if ref_words else None,
        **kwargs,
    )


def test_wer_is_micro_averaged_not_a_mean_of_rates():
    """Total errors over total words, which is what a corpus rate means."""
    rows = [
        scored("a", "en", 10, 1),   # 10%
        scored("a", "en", 100, 50),  # 50%
    ]
    summary = summarise("a", rows)
    assert summary.wer == pytest.approx(51 / 110)
    # A mean of the two rates would be 30%, which over-weights the tiny clip.
    assert summary.wer != pytest.approx(0.30)


def test_unsupported_clips_are_counted_separately_from_failures():
    rows = [
        scored("phonon", "en", 10, 0),
        ClipResult(
            clip_id="de-1", source="test", language="de", provider="phonon",
            status=UNSUPPORTED, note="en only",
        ),
    ]
    summary = summarise("phonon", rows)
    assert summary.clips_scored == 1
    assert summary.clips_skipped == 1
    assert summary.clips_failed == 0
    assert summary.skip_reasons == {"en only": 1}
    # The skipped clip contributes nothing, rather than counting as 100% error.
    assert summary.wer == 0.0


def test_failures_are_reported_but_do_not_skew_the_rate():
    rows = [
        scored("a", "en", 10, 1),
        ClipResult(
            clip_id="x", source="test", language="en", provider="a",
            status=ERROR, note="engine busy",
        ),
    ]
    summary = summarise("a", rows)
    assert summary.clips_failed == 1
    assert summary.wer == pytest.approx(0.1)


def test_no_scored_clips_gives_no_rate_rather_than_zero():
    summary = summarise("a", [])
    assert summary.wer is None
    assert summary.cer is None
    assert summary.clips_scored == 0


def test_per_language_breakdown_is_reported():
    rows = [
        scored("a", "en", 10, 1),
        scored("a", "de", 10, 5),
        scored("a", "de", 10, 5),
    ]
    summary = summarise("a", rows)
    assert summary.by_language["en"]["wer"] == pytest.approx(0.1)
    assert summary.by_language["de"]["wer"] == pytest.approx(0.5)
    assert summary.by_language["de"]["clips"] == 2


def test_latency_and_realtime_factor_are_averaged_over_what_was_measured():
    rows = [
        scored("a", "en", 10, 0, latency_ms=100.0, real_time_factor=0.1),
        scored("a", "en", 10, 0, latency_ms=300.0, real_time_factor=0.3),
        # No timing at all: must not pull the average toward zero.
        scored("a", "en", 10, 0),
    ]
    summary = summarise("a", rows)
    assert summary.mean_latency_ms == pytest.approx(200.0)
    assert summary.mean_real_time_factor == pytest.approx(0.2)


def test_summarise_all_keeps_provider_order():
    rows = [scored("b", "en", 10, 0), scored("a", "en", 10, 0)]
    summaries = summarise_all(rows, ["a", "b"])
    assert [summary.provider for summary in summaries] == ["a", "b"]


# -- reporting -------------------------------------------------------------


def test_tables_render_without_crashing():
    rows = [scored("a", "en", 10, 1), scored("a", "de", 10, 5)]
    summaries = summarise_all(rows, ["a"])
    assert "a" in render_table(summaries)
    language_table = render_language_table(summaries)
    assert "de" in language_table and "en" in language_table


def test_a_skipped_language_shows_a_dash_not_a_number():
    """'Not attempted' must not read as '100% wrong'."""
    rows = [scored("phonon", "en", 10, 0)]
    summaries = summarise_all([*rows], ["phonon"])
    summaries[0].by_language = {"en": summaries[0].by_language["en"], "de": None}
    table = render_language_table(summaries)
    assert "\u2014" in table


def test_csv_has_a_row_per_clip_and_the_expected_columns():
    rows = [scored("a", "en", 10, 1), scored("b", "en", 10, 0)]
    text = to_csv(rows)
    lines = text.strip().splitlines()
    assert len(lines) == 3  # header plus two rows
    assert lines[0].startswith("clip_id,source,language,provider")
    assert "wer" in lines[0]


def test_json_payload_round_trips_and_rounds_floats():
    rows = [scored("a", "en", 3, 1)]
    summaries = summarise_all(rows, ["a"])
    payload = to_json(summaries, rows, metadata={"clips": 1})
    encoded = json.dumps(payload)
    restored = json.loads(encoded)

    assert restored["metadata"]["clips"] == 1
    assert restored["summary"][0]["provider"] == "a"
    assert restored["clips"][0]["wer"] == pytest.approx(1 / 3, abs=1e-4)


def test_json_is_written_as_readable_utf8():
    clip = Clip(
        clip_id="pl-0", source="s", language="pl",
        reference="zażółć gęślą jaźń", path="p",
    )
    payload = to_json([], [ClipResult(clip_id=clip.clip_id, source="s",
                                      language="pl", provider="a", reference=clip.reference)])
    encoded = json.dumps(payload, ensure_ascii=False)
    assert "zażółć" in encoded
