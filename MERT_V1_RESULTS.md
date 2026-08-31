# MERT-v1 broad-genre development results

Development run: 2026-08-30. This is a development comparison, not a final
test result. The 90 previously observed test tracks and the 239 new holdout
tracks remain excluded; the run made zero test predictions.

## Question tested

Would the official `m-a-p/MERT-v1-95M` encoder improve the same four-label
classifier when every downstream choice and every data boundary remained fixed?

MERT-v1 was pretrained at 24 kHz on roughly 20,000 hours of music, whereas the
previous MERT-v0-public encoder used 16 kHz and roughly 900 hours. This
experiment changed the pretrained representation only. It reused the same 903
training tracks, 303 development-validation tracks, broad-label ontology,
artist-grouped folds, logistic-regression grid, and threshold rules. See
`experiments/MERT_V1_PROTOCOL.md` and the resampling correction in
`experiments/MERT_V1_IMPLEMENTATION_NOTE.md`.

## Integrity checks

- Official model revision: `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`.
- Model checkpoint SHA-256:
  `a2b8b747f72c06e0595aeae41ae5473f4364938c6b39b2c58be38c48e6bd3fcd`.
- Frozen v2 plan SHA-256:
  `2ceb8a71d8ce241f02b759620ca1b683847eaac3d2568616336f15624382a7c6`.
- Extracted feature SHA-256:
  `cd0b53a1dc54c00ae12a80a1cb94295de48773cdf2008d3600a16ff3f3f0b9f9`.
- Feature shape: 1,206 tracks x 13 layers x 768 values; all values finite.
- Test predictions: 0.

The first frozen plan failed before creating any features because its resampler
returned one extra sample. The corrected v2 implementation decodes exactly 30
source seconds and uses `soxr` HQ to produce exactly 720,000 samples at 24 kHz.
Both plans remain recorded so the correction is auditable.

## Selected heads

| Label | MERT-v1 representation | C | Class weight | Mean grouped-CV AP |
| --- | --- | ---: | --- | ---: |
| electronic | layer 8 | 0.001 | none | 0.7652 |
| pop | layer 6 | 0.001 | none | 0.5230 |
| ambient | layer 6 | 0.001 | balanced | 0.4519 |
| rock | layer 6 | 0.001 | balanced | 0.6276 |

Selections were made from training folds only. Development-validation outcomes
were not used to change the selected layer, regularization, or threshold.

## Development validation

The F1-oriented thresholds were 0.450, 0.325, 0.475, and 0.575.

| Label | Positives | Precision | Recall | F1 | AP | Predicted labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| electronic | 119 | 0.667 | 0.521 | 0.585 | 0.662 | 93 |
| pop | 49 | 0.385 | 0.306 | 0.341 | 0.352 | 39 |
| ambient | 70 | 0.387 | 0.657 | 0.487 | 0.452 | 119 |
| rock | 50 | 0.592 | 0.580 | 0.586 | 0.675 | 49 |

Micro precision was 0.507, micro recall 0.528, micro-F1 0.517, macro AP 0.535,
and output coverage was 235/303 = 0.776. The provisional usefulness target was
not met.

The precision-target rule was supported by training out-of-fold data only for
`electronic` (threshold 0.60). On validation it emitted 43 predictions with 33
true positives: precision 0.767 and recall 0.277. The other three labels were
suppressed, so this policy also failed the target.

## Comparison with MERT-v0-public

| Development metric | MERT-v0 | MERT-v1 | Change |
| --- | ---: | ---: | ---: |
| micro precision | 0.437 | 0.507 | +0.070 |
| micro recall | 0.590 | 0.528 | -0.063 |
| micro-F1 | 0.502 | 0.517 | +0.015 |
| macro AP | 0.524 | 0.535 | +0.012 |
| output coverage | 0.881 | 0.776 | -0.106 |

The representation helped overall precision and especially ambient ranking,
but the gains were modest and mixed. Electronic AP fell from 0.681 to 0.662;
pop AP was essentially unchanged; ambient AP rose from 0.387 to 0.452; rock AP
was essentially unchanged. The model still produces too many false positives
for pop and ambient.

## Decision

Do not promote MERT-v1 and do not open the holdout. More pretraining alone did
not solve the task. The next bounded candidate should use a representation
trained explicitly for music-style tagging, such as the Discogs-trained MAEST
model, while preserving the current development boundary. Its runtime,
taxonomy, model license, and Essentia dependency license must be checked before
committing to another full extraction run.
