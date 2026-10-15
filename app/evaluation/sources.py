"""A small, ground-truthed evaluation set, fetched on demand.

Both models publish numbers on the same public sets, so rather than inventing a
dataset this pulls a handful of real clips from those sets. It deliberately does
**not** download the corpora: LibriSpeech alone is 29 GB, and the full parquet
files for the Open ASR Leaderboard subsets are 300-500 MB each.

Instead it asks Hugging Face's datasets-server for one page of rows and downloads
only the audio it needs, so a useful, varied set costs a few megabytes.

Verified against the live API:
  * rows carry both a transcript and an audio URL;
  * the audio field is a list shaped ``[{src, type}]`` for both sources;
  * MLS audio is Ogg Opus, which libsndfile reads, and is already 16 kHz.

Sources are chosen to cover what the two models differ on: English read speech
(LibriSpeech clean and other) and the six non-English languages Whistle supports
but Phonon-2 does not.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100  # the datasets-server maximum
USER_AGENT = "voice-bench/0.1 (+local ASR evaluation)"
TIMEOUT_S = 180


@dataclass(frozen=True)
class Source:
    """One dataset config to draw clips from."""

    key: str
    repo: str
    config: str
    split: str
    #: Field holding the ground-truth transcript.
    text_field: str
    language: str
    note: str
    #: Offsets to probe, so clips come from different parts of the corpus.
    offsets: tuple[int, ...] = (0,)


#: Ordered so the canonical English material comes first.
CATALOG: tuple[Source, ...] = (
    Source(
        key="librispeech-clean",
        repo="openslr/librispeech_asr",
        config="clean",
        split="test",
        text_field="text",
        language="en",
        note="Read audiobook speech, studio quality. The common reference point.",
        offsets=(0, 400, 900, 1500),
    ),
    Source(
        key="librispeech-other",
        repo="openslr/librispeech_asr",
        config="other",
        split="test",
        text_field="text",
        language="en",
        note="Read speech that is harder: noisier, more accented, less clean.",
        offsets=(0, 300),
    ),
    Source(
        key="mls-german",
        repo="facebook/multilingual_librispeech",
        config="german",
        split="test",
        text_field="transcript",
        language="de",
        note="German read speech. Whistle supports this; Phonon-2 does not.",
        offsets=(0, 200),
    ),
    Source(
        key="mls-french",
        repo="facebook/multilingual_librispeech",
        config="french",
        split="test",
        text_field="transcript",
        language="fr",
        note="French read speech.",
        offsets=(0, 200),
    ),
    Source(
        key="mls-spanish",
        repo="facebook/multilingual_librispeech",
        config="spanish",
        split="test",
        text_field="transcript",
        language="es",
        note="Spanish read speech.",
        offsets=(0, 200),
    ),
    Source(
        key="mls-italian",
        repo="facebook/multilingual_librispeech",
        config="italian",
        split="test",
        text_field="transcript",
        language="it",
        note="Italian read speech.",
        offsets=(0, 200),
    ),
    Source(
        key="mls-dutch",
        repo="facebook/multilingual_librispeech",
        config="dutch",
        split="test",
        text_field="transcript",
        language="nl",
        note="Dutch read speech.",
        offsets=(0, 200),
    ),
    Source(
        key="mls-polish",
        repo="facebook/multilingual_librispeech",
        config="polish",
        split="test",
        text_field="transcript",
        language="pl",
        note="Polish read speech.",
        offsets=(0, 200),
    ),
)

CATALOG_BY_KEY = {source.key: source for source in CATALOG}

#: A small default set: two English sources plus every Whistle language.
DEFAULT_SOURCES: tuple[str, ...] = (
    "librispeech-clean",
    "librispeech-other",
    "mls-german",
    "mls-french",
    "mls-spanish",
    "mls-italian",
    "mls-dutch",
    "mls-polish",
)


@dataclass
class Clip:
    """One evaluation clip: audio on disk plus its ground truth."""

    clip_id: str
    source: str
    language: str
    reference: str
    path: str
    duration_s: float | None = None
    speaker_id: str | None = None
    note: str = ""
    origin: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# -- row access ------------------------------------------------------------


def fetch_rows(source: Source, offset: int, length: int = PAGE_SIZE) -> list[dict]:
    """Ask the datasets-server for a page of rows."""
    query = urllib.parse.urlencode(
        {
            "dataset": source.repo,
            "config": source.config,
            "split": source.split,
            "offset": offset,
            "length": length,
        }
    )
    request = urllib.request.Request(
        f"{ROWS_ENDPOINT}?{query}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        # Decode explicitly: the response omits a charset, and guessing latin-1
        # mangles accented transcripts.
        payload = json.loads(response.read().decode("utf-8"))
    return [entry.get("row") or {} for entry in payload.get("rows") or []]


def audio_url(row: dict) -> str | None:
    """Extract the audio URL, which arrives as a list of ``{src, type}``."""
    audio = row.get("audio")
    candidates: list[Any] = []
    if isinstance(audio, list):
        candidates = [item.get("src") for item in audio if isinstance(item, dict)]
    elif isinstance(audio, dict):
        candidates = [audio.get("src")]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.startswith("http"):
            return candidate
    return None


def text_of(row: dict, text_field: str) -> str:
    value = row.get(text_field)
    return value.strip() if isinstance(value, str) else ""


def select_varied_rows(
    rows: Iterable[dict],
    source: Source,
    want: int,
    seen_speakers: set[str] | None = None,
) -> list[dict]:
    """Pick rows, preferring a new speaker each time.

    Diversity of voice matters more than diversity of sentence for a model
    comparison: nine utterances from one speaker measure that speaker, not the
    model. Rows with no transcript or no audio are skipped rather than guessed at.
    """
    seen = seen_speakers if seen_speakers is not None else set()
    chosen: list[dict] = []
    fallback: list[dict] = []
    for row in rows:
        if not text_of(row, source.text_field) or not audio_url(row):
            continue
        speaker = str(row.get("speaker_id") or "")
        if speaker and speaker not in seen:
            seen.add(speaker)
            chosen.append(row)
        else:
            fallback.append(row)
        if len(chosen) >= want:
            return chosen
    # Not enough distinct speakers on this page: fill from the rest rather than
    # returning fewer clips than asked for.
    for row in fallback:
        if len(chosen) >= want:
            break
        chosen.append(row)
    return chosen


def probe_source(source: Source) -> dict[str, Any]:
    """Check a source is usable without downloading any audio."""
    rows = fetch_rows(source, source.offsets[0], length=5)
    if not rows:
        return {"source": source.key, "ok": False, "reason": "no rows returned"}
    usable = [row for row in rows if text_of(row, source.text_field) and audio_url(row)]
    if not usable:
        return {
            "source": source.key,
            "ok": False,
            "reason": f"rows carry no {source.text_field!r} or no audio",
        }
    return {
        "source": source.key,
        "ok": True,
        "language": source.language,
        "sample_reference": text_of(usable[0], source.text_field)[:70],
    }


# -- downloading -----------------------------------------------------------


def sniff_extension(data: bytes) -> str:
    """Name the container from its magic bytes.

    The datasets-server hands back signed URLs whose extension is not visible,
    and the corpora mix FLAC with Ogg Opus, so the bytes are the only reliable
    signal. It is cosmetic either way: the harness decodes with libsndfile and
    re-encodes to 16 kHz mono WAV.
    """
    if data[:4] == b"OggS":
        return ".opus"
    if data[:4] == b"fLaC":
        return ".flac"
    if data[:4] == b"RIFF":
        return ".wav"
    if data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3"):
        return ".mp3"
    return ".bin"


def download_clip(url: str, destination: Path) -> Path:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        data = response.read()
    target = destination.with_suffix(sniff_extension(data))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def fetch_clip(
    row: dict,
    source: Source,
    index: int,
    audio_dir: Path,
) -> Clip:
    """Download one row's audio and build its clip record."""
    url = audio_url(row)
    if url is None:
        raise ValueError("row has no audio URL")
    reference = text_of(row, source.text_field)
    if not reference:
        raise ValueError("row has no transcript")

    clip_id = f"{source.key}-{index:02d}"
    path = download_clip(url, audio_dir / clip_id)

    duration = row.get("audio_duration")
    if not isinstance(duration, (int, float)):
        duration = None
    return Clip(
        clip_id=clip_id,
        source=source.key,
        language=source.language,
        reference=reference,
        # Absolute, so the manifest keeps working when evaluated from a different
        # working directory.
        path=str(Path(path).resolve()),
        duration_s=round(float(duration), 2) if duration else None,
        speaker_id=str(row.get("speaker_id")) if row.get("speaker_id") else None,
        note=source.note,
        origin=f"{source.repo}/{source.config}/{source.split}",
    )


def fetch_rows_with_retry(
    source: Source, offset: int, attempts: int = 3, base_delay: float = 1.5
) -> list[dict]:
    """Fetch a page, retrying briefly.

    The datasets-server rate-limits anonymous callers, and it does so by failing
    the request rather than by a documented header, so a couple of short retries
    make a multi-source fetch behave far more predictably.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fetch_rows(source, offset)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last = exc
            if attempt < attempts - 1:
                time.sleep(base_delay * (attempt + 1))
    assert last is not None
    raise last


def usable_rows(rows: Iterable[dict], source: Source) -> list[dict]:
    return [row for row in rows if text_of(row, source.text_field) and audio_url(row)]


def distinct_speakers(rows: Iterable[dict], source: Source) -> set[str]:
    return {
        str(row.get("speaker_id"))
        for row in usable_rows(rows, source)
        if row.get("speaker_id")
    }


@dataclass
class FetchReport:
    clips: list[Clip] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.clips)


def fetch_set(
    source_keys: Iterable[str],
    output_dir: Path,
    per_source: int = 2,
    progress: Callable[[str], None] | None = None,
) -> FetchReport:
    """Fetch a small set of clips across the named sources."""
    audio_dir = Path(output_dir) / "audio"
    report = FetchReport()
    used_speakers: dict[str, set[str]] = {}

    for key in source_keys:
        source = CATALOG_BY_KEY.get(key)
        if source is None:
            report.failures.append(f"unknown source {key!r}")
            continue
        if progress:
            progress(f"{key}: fetching rows")

        seen = used_speakers.setdefault(source.key, set())

        # Read offsets until there are enough *distinct speakers*, not just enough
        # rows. A single page is 100 rows, so a row-count limit would only ever
        # read the first offset, and a page holding one speaker would supply every
        # clip from that one voice. Stopping on speaker count also keeps the
        # request count low, which matters because the datasets-server rate-limits
        # anonymous callers.
        pool: list[dict] = []
        for offset in source.offsets:
            try:
                pool.extend(fetch_rows_with_retry(source, offset))
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                report.failures.append(f"{key} offset {offset}: {exc}")
            if len(distinct_speakers(pool, source) | seen) >= per_source:
                break

        chosen = select_varied_rows(pool, source, per_source, seen) if pool else []

        if not chosen:
            report.failures.append(f"{key}: no usable rows found")
            continue

        for index, row in enumerate(chosen[:per_source]):
            try:
                clip = fetch_clip(row, source, index, audio_dir)
            except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
                report.failures.append(f"{key} clip {index}: {exc}")
                continue
            report.clips.append(clip)
            if progress:                progress(
                    f"{key}: {clip.clip_id} "
                    f"{clip.duration_s or '?'}s speaker={clip.speaker_id} "
                    f"{(clip.reference[:44] + '...') if len(clip.reference) > 44 else clip.reference}"
                )

    return report


def write_manifest(clips: list[Clip], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for clip in clips:
            handle.write(json.dumps(clip.as_dict(), ensure_ascii=False) + "\n")
    return path


def read_manifest(path: Path) -> list[Clip]:
    clips: list[Clip] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            clips.append(Clip(**json.loads(line)))
    return clips
