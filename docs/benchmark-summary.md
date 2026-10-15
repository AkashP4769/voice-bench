# Benchmark summary: Phonon-2 vs Whistle

Short conclusion for the latest stored run. The long-form version, with the earlier
multilingual set, is in [benchmark-conclusion.md](./benchmark-conclusion.md).

| | |
| --- | --- |
| Run | `20261015-221705` — `data/benchmarks/librispeech/runs/20261015-221705.json` (+ `.csv`) |
| Recorded | 2026-10-15 22:17 |
| Set | 20 LibriSpeech test clips (10 `clean`, 10 `other`), 160 s of audio, 431 reference words |
| Models | `phonon-2`, `whistle` — both in-process, both scored on **identical** audio |
| Scoring | `lower_strip_punctuation` |
| Machine | AMD Ryzen 7 4800H, Python 3.10.6 |

## Result

| Model | WER | CER | Perfect clips | Mean latency | RTF |
| --- | --- | --- | --- | --- | --- |
| `phonon-2` | **1.62%** | **0.57%** | 15/20 | **407 ms** | **0.056** |
| `whistle` | 5.10% | 3.44% | 9/20 | 691 ms | 0.075 |

| Subset | Phonon-2 | Whistle |
| --- | --- | --- |
| `clean` (260 words) | **0.00%** (0 errors) | 4.23% (11 errors) |
| `other` (171 words) | **4.09%** (7 errors) | 6.43% (11 errors) |

Error mix: Phonon-2 made 6 substitutions, 0 insertions, 1 deletion. Whistle made
16 substitutions, 1 insertion, 5 deletions.

## Findings

1. **Phonon-2 wins on English read speech**, and on the clean subset it made no
   errors at all across 260 words.
2. **The gap narrows on harder audio** (4.09% vs 6.43%), which is where the two
   are closest.
3. **Phonon-2 is ~1.7× faster** on the same clips (407 ms vs 691 ms; RTF 0.056 vs
   0.075). Both are far below real time.
4. **Whistle loses about a fifth of its errors to one formatting artifact.**
   On `librispeech-clean-02` it wrote `"On August 27, 1837, she writes."` against a
   reference of `"ON AUGUST TWENTY SEVENTH EIGHTEEN THIRTY SEVEN SHE WRITES"` — it
   transcribed the date correctly and then scored 55.56%, because the policy does
   not expand numerals. Excluding that clip, Whistle is 4.03% rather than 5.10%.
5. **Both models share one failure**: `librispeech-other-00`, where the reference
   spells out `"I AM ..."` and both wrote `"I'm ..."` (22.22% each). Apostrophes
   are deliberately kept by the normalizer, so a contraction against expanded
   reference words counts as an error even though nothing was misheard.
6. **The harness is repeatable**: Phonon-2 scored exactly 1.62% WER / 0.57% CER in
   both the 22:15 and 22:17 runs, so run-to-run noise is below the differences
   reported here.

## Verdict

For English, Phonon-2 is the better model here: more accurate, faster, and it
handles numbers and contractions the way the references are written. Whistle's
real gap is smaller than 5.10% suggests once numeral formatting is discounted,
but it does not beat Phonon-2 on any subset in this run.

## Caveats

- 20 clips from one corpus, English only — enough to rank these two models on read
  speech, not enough to call it a general result.
- The findings in (4) and (5) are scoring-policy artifacts, not recognition
  errors. Expanding numerals and contractions before scoring would raise both
  models' accuracy, Whistle's more than Phonon-2's.
- Beyond that, `clean` and `other` are both audiobook-derived, so neither tests
  conversational, noisy or far-field speech.
