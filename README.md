# Voice Bench

A local benchmark harness for two on-device speech recognition models:

- **Phonon-2** — Fermion Research, via `fermion-research`
- **Whistle** — Cactus Compute, via `cactus-needle`

It runs both models on the same normalized audio, reports accuracy and latency
side by side, and stores every run with the environment that produced it.

Neither vendor offers a hosted transcription API today, so **both models run
locally**. The architecture still assumes a hosted provider will arrive: every
provider is reached through a transport, and one of those transports speaks the
OpenAI-compatible `POST /v1/audio/transcriptions` contract. Switching a provider
from local to hosted is a change to `providers.yaml`, not to any code.

---

## Quick start

```bash
python -m venv venv
.\venv\Scripts\activate            # Windows
# source venv/bin/activate         # macOS / Linux

pip install -e ".[engines,dev]"    # both models, plus test tooling
```

Installing only what you need is supported, because the two engines have very
different weight:

```bash
pip install -e ".[whistle]"        # cactus-needle only, no torch
pip install -e ".[phonon]"         # fermion-research, pulls in torch
```

Load the models once so a first-run download never pollutes a timing:

```bash
bench warmup
```

### Web UI

```bash
bench serve            # http://127.0.0.1:8080
```

Upload a clip or record one from the microphone, pick the models, optionally
paste a reference transcript, and compare. The page also lists each provider's
declared capabilities, so it is clear up front that Phonon-2 is English-only and
cannot select a language.

### Recording from the microphone

The **Audio source** row offers two ways in: *Choose audio file*, or the
microphone button, which records and then attaches the take to the same form, so
every other setting applies unchanged. While recording you get a live timer,
an input level meter, and Stop/Discard; the Run button is disabled until the take
is ready, and a finished recording can be transposed or discarded and re-recorded.

Two implementation notes, both of which matter:

- **Raw PCM, not `MediaRecorder`.** `MediaRecorder` emits webm/opus (or mp4/aac),
  and the backend decodes audio with libsndfile, which reads wav, flac, ogg and
  aiff but not webm or mp4. Fermion's server rejects mp3 and m4a outright. So the
  Web Audio graph captures raw samples in an `AudioWorklet` and encodes a 16 kHz
  mono 16-bit PCM WAV in the browser — the format both engines natively want.
- **A secure context is required.** `getUserMedia` is unavailable on plain
  `http://` except on `localhost`, so the app reports that rather than failing
  silently. If the page is opened from a LAN address, use the file picker.

Recordings are named `recording-<timestamp>.wav`, saved under `data/audio/uploads`,
and normalized and hashed like any other upload, so a recorded run is as
reproducible as a file-based one. Recordings are capped at 5 minutes.

### Command line

```bash
bench providers

bench run tests/fixtures/short_en.wav \
      --providers whistle,phonon-2 \
      --reference-text "The quick brown fox jumps over the lazy dog." \
      --json-out run.json --csv-out run.csv
```

```
provider     status   transport      latency      ttft     rtf     wer     cer words
--------------------------------------------------------------------------------------
whistle      ok       in_process      598 ms    53 ms   0.177   0.00%   0.00%     9
phonon-2     ok       in_process      296 ms        —   0.088   0.00%   0.00%     9
```

Numbers are from one Windows machine and are examples, not claims about the
models in general. What the harness promises is that the two rows were produced
from byte-identical audio with recorded settings.

---

## How it fits together

```text
providers.yaml -> provider registry -> adapter -> transport
                                                  |- in_process    (engine in a worker process)
                                                  +- openai_http   (local server, or hosted + API key)
```

Four layers, each replaceable on its own:

| Layer | Responsibility | Where |
| --- | --- | --- |
| Domain | `TranscriptionResult`, `TranscriptionOptions`, capabilities | `app/domain/` |
| Adapter | One file per model; maps engine output into the domain shape | `app/providers/` |
| Transport | How a provider is reached | `app/transport/` |
| Workers | One process per provider, one request at a time | `app/workers/` |

### Why one process per provider

Both engines allow exactly one model per process, and neither is thread-safe:

- `needle`: *"One model is loaded per process; it is not thread-safe."*
- Fermion: one model behind a single worker thread, every decode serialized
  behind one lock.

So a worker process owns each model for its lifetime, and the pool serializes
requests per provider. This also keeps the heavy `torch` import out of the API
process and contains a crashing engine.

Worker control traffic runs over a loopback socket, not stdout. Both engines
print progress to stdout, and redirecting file descriptors around a native
library proved unreliable, so the two channels are kept separate by
construction. Engine output goes to `data/logs/worker-<name>.log`.

---

## Fair comparison

A benchmark is only useful if the comparison is honest. What that requires here:

- **Identical bytes.** Audio is decoded once to 16 kHz mono float32, and every
  provider receives that same normalized file. Its SHA-256 is recorded.
- **The transport is recorded.** An in-process decode and a round trip over HTTP
  are not the same measurement. A run that mixes them is flagged as not
  directly comparable, and the mismatch is listed in the run's caveats.
- **No invented numbers.** A provider that cannot measure time-to-first-token
  reports nothing rather than zero. Absent means absent, in the UI and the JSON.
- **Normalization is declared.** Word error rate is computed after a named
  normalization policy, and the policy is stored with the run. The default is
  `lower_strip_punctuation`.
- **Engine-reported caveats surface.** Phonon-2 reports `truncated` when a window
  exhausted its token budget and audio may be missing; that becomes a visible
  warning on the result, not a footnote.
- **Capabilities gate the UI.** A provider that cannot honour an option is not
  silently asked to ignore it. It returns an error for that provider only, and
  the model is never loaded in the first place.

### Word error rate is the harness's own

`wer`/`cer` come from an edit-distance implementation in `app/scoring/metrics.py`
with an explicit normalization policy. This is **not** Whisper's
`EnglishTextNormalizer`: that one expands numerals and applies a long
substitution list, and claiming parity without porting it would be misleading.
Every run says which policy was used.

---

## Metrics

| Metric | Meaning |
| --- | --- |
| `wer` | Word error rate against the reference, after normalization |
| `cer` | Character error rate, whitespace excluded |
| `substitutions` / `insertions` / `deletions` | How the errors break down |
| `latency_ms` | Wall time of the provider call, excluding model load |
| `time_to_first_token_ms` | Engine-reported, where the engine reports it |
| `real_time_factor` | `latency / audio duration`; below 1 is faster than real time |
| `window_count` | Windows the **harness** cut the audio into |
| `engine_window_count` | Windows the **engine** used internally |

Model load time is deliberately excluded from latency: `warmup()` runs before
the first job, and the measured load time is reported separately.

---

## Configuration

`providers.yaml` defines the providers. `api_key_env` holds the **name** of an
environment variable, never a secret value, and no endpoint ever echoes a key.

```yaml
providers:
  phonon-2:
    type: phonon
    model: phonon-2
    transport: in_process
    server:
      base_url: http://127.0.0.1:8000
      api_key_env: PHONON_API_KEY
```

### Reaching a model over HTTP instead

Fermion ships an OpenAI-compatible server, so the seam can be exercised today:

```bash
fermion serve phonon-2 --port 8000
```

```yaml
  phonon-2-http:
    type: phonon
    model: phonon-2
    transport: openai_http
    server:
      base_url: http://127.0.0.1:8000
      api_key_env: PHONON_API_KEY
```

Verified: the in-process and HTTP paths return the **same text and the same
word timings** for the same clip.

### Reaching a hosted provider

Same change, different values. The key stays in the environment:

```bash
setx HOSTED_SPEECH_KEY "..."     # PowerShell: $env:HOSTED_SPEECH_KEY = "..."
```

```yaml
  hosted:
    type: phonon
    model: phonon-2
    transport: openai_http
    server:
      base_url: https://api.example.com/v1
      api_key_env: HOSTED_SPEECH_KEY
      timeout_s: 300
```

A loopback `base_url` needs no key. A remote one reports
*configured / not configured* without ever reading the value.

---

## Adding a model

1. Write an adapter in `app/providers/<name>.py` subclassing `BaseProvider`:
   declare `capabilities`, implement `transcribe_local()` and, if the engine
   needs setup, `warmup()`.
2. Register the type in `ADAPTERS` in `app/providers/registry.py`.
3. Add an entry to `providers.yaml`.

Nothing in the API, the UI, the scorer, the storage layer or the worker pool
changes. Adapters import their engine lazily, so a machine that has one engine
installed never trips over the other's imports.

---

## CLI

| Command | Purpose |
| --- | --- |
| `bench providers` | List providers, transports and declared capabilities |
| `bench warmup [--provider NAME]` | Download and load models before timing anything |
| `bench run AUDIO --providers a,b` | Benchmark one file; `--reference`, `--json-out`, `--csv-out` |
| `bench list` / `bench show ID` | Inspect stored runs |
| `bench env` | Machine, package versions, and Phonon's CPU kernel tier |
| `bench serve` | Web UI and API |

## API

| Route | Purpose |
| --- | --- |
| `GET /api/health` | Service status and worker state |
| `GET /api/providers` | Capabilities, transport, credential state (never a key) |
| `POST /api/providers/{p}/warm` | Load one model now |
| `POST /api/runs` | Multipart: `file` or `audio_path`, `providers`, `language`, `keywords`, `reference_text`, `notes`, `normalization` |
| `GET /api/runs` | Recent runs |
| `GET /api/runs/{id}` | One run with normalized results |
| `GET /api/runs/{id}/export?format=csv\|json` | Machine-readable export |
| `DELETE /api/runs/{id}` | Delete a run |
| `GET /` | Web UI |

---

## Tests

```bash
pytest -m "not integration"    # fast: no model is loaded
pytest -m integration          # loads both real models
pytest                         # everything
```

The fast suite replaces the worker pool with a stub, so it exercises upload,
normalization, capability checks, scoring, persistence, export and rendering
without loading a model. The integration suite is what proves both adapters
agree on the normalized contract, including word timings that stay ordered and
inside the audio.

Speech fixtures live in `tests/fixtures/`. They are generated with Windows SAPI
at 16 kHz mono, which conveniently yields a known reference transcript:

```powershell
powershell -File tools/make_fixtures.ps1
```

Unit tests that need audio generate their own synthetic clips, so the fast suite
runs on any platform.

---

## Notes and limits

- **Phonon-2 is English-only** and its server accepts but ignores a `language`
  field, so language selection is not offered for it.
- **Both engines segment long audio themselves** (verified: Whistle transcribed a
  47 s clip complete in a single call). The harness has chunking and stitching
  available for providers that declare a `max_duration_s`, but neither of these
  uses it, so no result is confounded by our own windowing.
- **No Docker.** Both engines install natively on Windows, Linux and macOS, so
  containers would add a boundary without adding capability. They remain a
  sensible escape hatch for CUDA, for reproducing a specific CPU kernel tier, and
  for CI. If one is ever needed, it is one container for the backend plus
  optional inference containers reached through the same HTTP transport — never
  one container per model.
- **Requests are synchronous.** A long file occupies a request thread. The run
  is already modelled as a job (a run row plus one row per provider), so moving
  it to a background queue later does not change the API.
- **Windows and `torch`.** A clean machine may need Microsoft's
  `vc_redist.x64.exe`; without it `import torch` fails with `WinError 126`. The
  adapter detects this and reports the fix instead of a traceback.
- **Telemetry.** `cactus-needle` sends anonymous usage counts by default. The
  adapter sets `NEEDLE_TELEMETRY=0`, because locally benchmarked audio is the
  user's business.
- **Microphone audio never leaves the machine**, and it is not sent anywhere
  until the recording has been attached and you press Run benchmark. The browser
  asks for permission the first time; Discard releases the microphone
  immediately (and the tracks are stopped, so the browser's recording indicator
  clears).
- **Static assets are versioned** (`/static/app.css?v=<version>`). Without that,
  a browser will happily serve a cached stylesheet after an update, which makes
  a CSS change look like it did nothing. Bump `__version__` if you need to force
  a refresh during development.
