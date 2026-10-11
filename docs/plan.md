# Local Speech-to-Text Benchmark Harness (Phonon-2 vs Whistle)

## Problem

Benchmark two on-device speech-to-text models against each other on local hardware:

- **Phonon-2** — Fermion Research (`pip install fermion-research`)
- **Whistle** — Cactus Compute (`pip install cactus-needle`)

Neither vendor offers a hosted API-key service, so **the MVP is local-only**. However, the
architecture must accept a future hosted provider that authenticates with an API key,
without a rewrite.

## Key design bet

A single seam: an **OpenAI-compatible `/v1/audio/transcriptions` transport**.

Phonon already ships exactly this endpoint locally (`fermion serve phonon-2`). Building the
transport against that contract means "local now, hosted later" is a configuration change
(`base_url` + `api_key_env`), not a new code path.

```text
providers.yaml -> registry -> SpeechProvider adapter -> transport
                                                        |- in_process     (fastest, Python API)
                                                        |- openai_http    (local server)
                                                        +- openai_http    (hosted + API key, future)
```

## Verified facts driving the design

| Fact | Consequence |
| --- | --- |
| Both ship pure-Python wheels (`py3-none-any`), Windows x64 supported | No Docker needed for the MVP; venv is enough |
| `fermion-research` depends on `torch` + `transformers` (~2.5 GB) | Heaviest dependency; lazy-import so Whistle-only users aren't blocked |
| `cactus-needle` has **no** torch dependency | Light install; fetches `windows-x86_64` engine + `whistle.cact` on first use |
| Phonon-2 is **English-only**, 16 kHz, auto-segments long audio | Capability flag; UI must not offer language selection for it |
| Whistle supports en/de/fr/es/it/nl/pl | Capability flag; language list differs per provider |
| `needle.transcribe()` accepts **at most 30 s** | Harness must window long audio; record the strategy (it affects WER) |
| Both engines: **one model per process, not thread-safe** | Process-per-provider worker; never share a model across threads |
| Phonon server is explicitly **one request at a time** | Same conclusion: serialize per provider, concurrency is our job |
| Cactus caches to `~/.cache/cactus-needle/v3/`; Phonon to `~/.cache/fermion` (or `FERMION_CACHE_DIR`) | Add an explicit warm-up step so first-run download isn't mistaken for inference latency |
| Windows may need `vc_redist.x64.exe` (`WinError 126` on `import torch`) | Surface the actionable install message instead of a traceback |

## Scope

**In scope (MVP)**

- Local inference for Phonon-2 and Whistle.
- Single audio file → run through one or more providers → side-by-side results.
- Normalized result shape across providers.
- Word timestamps where available.
- Optional reference transcript → WER/CER.
- Latency, real-time factor, first-token latency.
- Run persistence and reproducibility metadata.
- Web UI: upload, provider/option selection, results.
- CLI entry point for scripted/batch benchmarking.

**Out of scope (explicitly deferred)**

- Hosted API-key providers (seam is built, no implementation).
- Microphone/live streaming.
- Multi-user auth, deployment, scaling.
- GPU/CUDA paths (CPU first; RTF reporting makes GPU comparison a later addition).
- Fine-tuning or training.

## Repository structure

```text
app/
  __init__.py
  config.py                 # pydantic-settings; provider config + api_key_env indirection
  domain/
    models.py               # AudioClip, TranscriptionOptions, TranscriptionResult, RunRecord
    provider.py             # SpeechProvider protocol + ProviderCapabilities
  audio/
    prepare.py              # decode -> 16 kHz mono float32, hash, cache
    windowing.py            # 30 s chunking + overlap for Whistle; stitching -> absolute times
  transport/
    in_process.py           # calls the provider Python API directly
    openai_http.py          # POST /v1/audio/transcriptions -> hosted-API seam
  providers/
    registry.py             # load providers.yaml, instantiate adapters
    phonon.py
    whistle.py
  workers/
    pool.py                 # one process per provider, serialized jobs
  scoring/
    normalize.py            # whisper-style text normalization
    metrics.py              # WER, CER
  storage/
    db.py                   # SQLite via SQLModel/SQLAlchemy
    repository.py
  api/
    routes_providers.py
    routes_runs.py
    routes_health.py
  ui/
    templates/              # Jinja2 + HTMX
    static/
  main.py                   # FastAPI app factory
  cli.py                    # `bench` CLI
providers.yaml              # provider definitions (no secrets, only env var names)
data/
  audio/                    # uploaded + normalized audio (gitignored)
  results/                  # exported JSON/CSV (gitignored)
tests/
  fixtures/                 # small 16 kHz mono WAVs
pyproject.toml
README.md
```

## Core contracts

```python
@dataclass(frozen=True)
class ProviderCapabilities:
    languages: list[str] | None        # None = multilingual/auto-detect
    word_timestamps: bool
    hotwords: bool
    embeddings: bool
    max_duration_s: float | None       # Harness windows above this
    streaming: bool

class SpeechProvider(Protocol):
    name: str
    capabilities: ProviderCapabilities

    def transcribe(
        self,
        audio: PreparedAudio,
        options: TranscriptionOptions,
    ) -> TranscriptionResult: ...
```

`TranscriptionResult` normalizes provider differences (mirrors the providers' own honest output):

- `text`, `language`
- `words[]` — `{text, start_ms, end_ms, probability}`
- `audio_duration_ms`, `latency_ms`
- `time_to_first_token_ms` (Whistle reports `ttft_ms`; Phonon leaves it `None`)
- `real_time_factor`
- `raw_response` — untrimmed provider payload for debugging
- `windowing` — `{strategy, window_count}` when the harness chunked the audio

`TranscriptionResult` must never fabricate a field: absent metrics are `None`, not `0`.

## Key decisions

1. **One seam, two transports.** `in_process` for speed and fidelity; `openai_http` for
   parity and the future hosted path. Phonon has a native server → use it to prove the seam.
2. **The transport is recorded on every run.** Mixing in-process against HTTP silently
   invalidates a latency comparison, so the UI must show it.
3. **Serialized execution per provider.** One worker process per provider, jobs queued.
   Never `asyncio.gather` two models — it corrupts latency and violates single-model-per-process.
4. **Declared capabilities gate the UI.** The provider list drives which options are
   enabled, so Phonon's English-only limit and Whistle's 30 s cap can't be violated silently.
5. **Comparisons default to like-for-like.** Both providers run through the same transport in
   a comparison run; overrides are possible but flagged in the results.
6. **Normalize audio once.** Decode → 16 kHz mono float32 → hash → cache. All providers read
   the identical bytes; the hash is the join key for fairness and reproducibility.
7. **Windowing is explicit and recorded.** Our own 30 s chunking for Whistle (with overlap),
   stitched back to absolute times. Strategy is stored because it is a confounder for WER.
8. **Secrets by indirection only.** Config holds `api_key_env: PHONON_API_KEY`, never a value.
   The DB never stores a key. UI renders *configured / not configured* only.
9. **Explicit warm-up step.** `bench warmup` fetches engine + weights so the first timed run
   is real inference.
10. **Docker: not for the MVP.** Both run natively on this machine, so the venv is fastest.
    Containers remain an escape hatch for CUDA, reproducible CPU-tier comparison, and CI —
    and never one container per model. The future shape is one backend container plus
    *optional* inference containers reached through the same HTTP seam.
11. **Frontend: FastAPI + Jinja2 + HTMX.** Per your choice — no Node toolchain; enough for
    upload, tables, and side-by-side transcripts without a build step.

## Configuration (`providers.yaml`)

```yaml
providers:
  phonon-2:
    type: phonon
    model: phonon-2
    transport: in_process          # or openai_http
    server:
      base_url: http://127.0.0.1:8000
      api_key_env: PHONON_API_KEY  # name only; unused while local
  whistle:
    type: whistle
    transport: in_process
    server:
      base_url: http://127.0.0.1:8001
      api_key_env: WHISTLE_API_KEY
```

Adding a hosted provider later is a new YAML entry plus, at most, a new transport.

## API surface

```text
GET  /api/health
GET  /api/providers                 # capabilities + configured/unconfigured
POST /api/runs                      # {audio_file_id, providers[], options, reference?}
GET  /api/runs                      # history
GET  /api/runs/{run_id}             # status + normalized results + metrics
GET  /api/runs/{run_id}/export      # JSON / CSV
GET  /                              # HTMX UI
```

Job-oriented from the start (a `runs` row plus per-provider child results) so long audio and
batch work move to a background queue later without changing the API.

## Reproducibility record (per run)

Model revision, provider package version, transport, options hash, normalized-audio SHA-256,
audio duration, windowing strategy, CPU feature tier (`fermion describe`), thread count,
and timestamp. Without this, a benchmark result is not defensible.

## Build phases

1. **Scaffold + skeleton** — `pyproject.toml`, config, domain models, `providers.yaml`,
   capabilities registry. Whistle in-process adapter, called from the CLI. Proves the contract.
2. **Audio pipeline** — 16 kHz mono normalization, hashing, cache, verification fixtures.
3. **Windowing** — 30 s chunking with overlap for Whistle, stitching to absolute timestamps.
4. **Phonon adapter** — in-process `fermion.load_speech()`, with lazy import and a clear
   failure message when `torch` is unavailable.
5. **Workers** — process-per-provider pool, serialized queue, timeouts, crash isolation.
6. **Scoring** — normalization policy + WER/CER + reference transcript support.
7. **Persistence** — SQLite schema, repository, artifact storage.
8. **FastAPI + HTMX UI** — upload, provider/option pickers gated by capabilities, run,
   side-by-side transcripts, metrics table, word-timestamp view, export.
9. **HTTP transport** — generic OpenAI-compatible client; wire Phonon's local server through
   it and assert parity against `in_process`.
10. **Polish** — `bench warmup`, CLI batch mode, README, graceful degradation when a
    provider's deps are missing.

## Validation

- Unit: text normalization, windowing/stitching boundaries, WER/CER against known pairs.
- Contract: both providers return the same normalized shape for one fixture; word timestamps
  monotonic and within `audio_duration_ms`.
- Parity: `in_process` vs `openai_http` produce identical text for the same clip.
- Security: `/api/providers` reports configured state and never echoes a key value.
- Robustness: missing `torch` yields the actionable install message, not a traceback.
- End-to-end: fixture WAV → both providers → results rendered in the UI.

## Notes and risks

- Whistle's 30 s cap means long-audio WER partly reflects *our* windowing, not the model.
  Keep the window/overlap configurable and always report it.
- Phonon's first run downloads 164 MB; Whistle's downloads a 16.9 MB `.cact` plus an engine.
  Warm-up exists so this never pollutes a timed run.
- `needle.Whistle()` and Phonon's speech object are one-per-process — the pool must own model
  lifetime, not create models per request.
- A future `faster-whisper` provider is planned to prove the abstraction, since Whistle's own
  published numbers compare against Whisper.
