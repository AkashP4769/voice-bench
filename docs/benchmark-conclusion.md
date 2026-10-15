# Benchmark conclusion: Phonon-2 vs Whistle

Conclusion drawn from the latest **persisted** evaluation result in this repository.

## What was evaluated

| Item | Value |
| --- | --- |
| Result artifacts | [results.json](../data/benchmark/results.json), [results.csv](../data/benchmark/results.csv) |
| Clip set | [manifest.jsonl](../data/benchmark/manifest.jsonl) |
| Clips | 16 (191.6 s of audio) |
| Languages | `en`, `de`, `fr`, `es`, `it`, `nl`, `pl` |
| Models | `phonon-2`, `whistle` |
| Normalization policy | `lower_strip_punctuation` |
| Transport | `in_process` for both models |
| Clip/provider pairs | 32 total: 20 scored, 12 skipped, 0 failed |
| Run recorded | 2026-10-15 20:55 (local file timestamps) |
| Machine | AMD Ryzen 7 4800H (16 logical cores), 15.4 GB RAM, Python 3.10.6, Windows |

Sources are LibriSpeech `clean` and `other` test subsets plus Multilingual LibriSpeech German, French, Spanish, Italian, Dutch and Polish.

## Headline numbers

| Model | Clips scored | Clips skipped | WER | CER | Mean latency | Mean RTF |
| --- | --- | --- | --- | --- | --- | --- |
| `phonon-2` | 4 | 12 | **4.05%** | 0.93% | 383.7 ms | 0.061 |
| `whistle` | 16 | 0 | **18.67%** | 5.69% | 1525.6 ms | 0.113 |

These two rates are **not comparable as printed**: Phonon-2 never attempted 12 of the 16 clips, so each model is averaged over a different, differently weighted corpus. The comparable subset is below.

## The only fair comparison: the 4 shared English clips

| Model | Audio | Reference words | Errors | WER | Mean latency | Mean RTF |
| --- | --- | --- | --- | --- | --- | --- |
| `phonon-2` | 26.5 s | 74 | 3 | **4.05%** | **383.7 ms** | **0.061** |
| `whistle` | 26.5 s | 74 | 4 | 5.41% | 735.8 ms | 0.116 |

Per clip:

| Clip | Subset | Phonon-2 WER | Whistle WER |
| --- | --- | --- | --- |
| `librispeech-clean-00` | clean | 0.00% | 0.00% |
| `librispeech-clean-01` | clean | 0.00% | 5.26% |
| `librispeech-other-00` | other | 22.22% | 22.22% |
| `librispeech-other-01` | other | 2.63% | 2.63% |

By English subset (same clips for both models):

| Subset | Phonon-2 | Whistle |
| --- | --- | --- |
| `librispeech-clean` | 0.00% (0/27) | 3.70% (1/27) |
| `librispeech-other` | 6.38% (3/47) | 6.38% (3/47) |

## Whistle's multilingual behaviour

Phonon-2 declares English only, so every non-English clip was recorded as `unsupported` (`en only`) rather than scored. That is a capability boundary, not a failure.

| Language | Clips | Reference words | WER |
| --- | --- | --- | --- |
| `en` | 4 | 74 | 5.41% |
| `es` | 2 | 74 | 5.41% |
| `fr` | 2 | 70 | 18.57% |
| `nl` | 2 | 81 | 24.69% |
| `it` | 2 | 68 | 25.00% |
| `pl` | 2 | 63 | 25.40% |
| `de` | 2 | 68 | 27.94% |

## Error shape

| Model | Substitutions | Insertions | Deletions | Total errors |
| --- | --- | --- | --- | --- |
| `phonon-2` | 2 | 0 | 1 | 3 |
| `whistle` | 73 | 9 | 11 | 93 |

Whistle's errors are dominated by substitutions, which is what its transcripts show: it keeps fluent sentence structure but swaps individual words, especially proper nouns and low-frequency terms.

- `librispeech-clean-01`: "Hawkeye" became "Hakai" — the single error in that clip.
- `mls-german-00`: "zaubern" rendered as "herzalben", "Münchhausen" as "Mensch hausen".
- `mls-dutch-01`: "uitgenoodigd" rendered as "uitgenodigd" and several phrases rephrased rather than mistyped.

Phonon-2's three errors are also substitutions plus one deletion, and it matched whistle exactly on `librispeech-other-00` and `librispeech-other-01`.

## Conclusion

1. **On clean English read speech, Phonon-2 is the stronger model in this run** (0.00% vs 3.70% WER), and on the harder `other` subset the two are indistinguishable (6.38% each).
2. **Phonon-2 is roughly 1.9× faster than Whistle on identical audio** (383.7 ms vs 735.8 ms mean latency; RTF 0.061 vs 0.116). Both are well below real time on this CPU.
3. **Whistle is the only one of the two usable beyond English.** It matches its English score exactly on Spanish (5.41% for both), degrades to 18.57% on French, and lands near 25–28% on German, Dutch, Italian and Polish.
4. **Neither model failed a single clip** (0 failures across 20 scored pairs), and the harness's capability gating prevented 12 meaningless English-only comparisons instead of reporting them as ~100% error.

### Confidence and limits

This is a **directional** result, not a final one:

- The English head-to-head rests on **4 clips and 74 reference words**. A three-error difference between the models is within the noise of a set that small.
- Each non-English language is represented by only **2 clips**, so the multilanguage ranking (German worst, Spanish best) is weakly supported.
- Scoring uses the harness's own normalizer and edit-distance implementation, with `lower_strip_punctuation` applied to both sides. It is not Whisper's `EnglishTextNormalizer`.
- Multilingual references are unpunctuated MLS transcripts, while the models emit punctuated output; normalization removes that difference, but it means CER is the more sensitive of the two metrics here.
- Mean latency and RTF are averaged **over each model's own scored clips**, so the whole-set figures are not a like-for-like timing comparison — use the 4-clip table for that.

### To make this conclusive

- Grow the English comparison to at least ~50 clips from several speakers before declaring a winner.
- Add ~10 clips per non-English language to separate Whistle's language ranking from sampling noise.
- Re-run against the same manifest (`bench evaluate --manifest data/benchmark/manifest.jsonl`) after any model, package or CPU-kernel change, so numbers stay comparable.

## Data that is not in this conclusion

The results above came from the CLI, which persisted them with `--json-out` and
`--csv-out`. The automated benchmark page did not store its results at the time,
so a newer LibriSpeech-only run (20 clips, clean + other, fetched at 2026-10-15
21:48, 4 clips normalized at 21:49) has no result file and cannot be analysed
here.

That gap is now closed: every automated run is written to
`data/benchmarks/<dataset>/runs/<run-id>.json` and `.csv` before the page renders
it, with `latest.json` and `latest.csv` tracking the newest result. Re-running
that LibriSpeech set will therefore produce a stored result that can be compared
with this one.
