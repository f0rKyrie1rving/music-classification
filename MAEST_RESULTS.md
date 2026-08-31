# Discogs-MAEST broad-genre development results

Development run: 2026-08-30. This is a development result, not a final test
score. The 90 historical test tracks and 239 new holdout tracks remain excluded;
the run made zero test predictions.

## What changed

This experiment replaced the general MERT representation with the official
`mtg-upf/discogs-maest-30s-pw-129e-519l` model, which was pretrained to predict
519 Discogs music styles. It kept the same 903 training tracks, 303
development-validation tracks, broad-label ontology, grouped folds, six
logistic settings, and training-OOF threshold rules.

The model was pinned to revision
`6c35f32a350f74351870937d5ae0bae1d898d1df`. The safetensors SHA-256 is
`881cec6abdcb6ef986367e6c0db02dc760cc0f88a52b625cb9d6e2b54d505548`.
The reviewed local feature extractor uses the official fixed mel settings. The
single representation concatenates the seventh transformer block's CLS token,
DIST token, and mean of the remaining signal tokens into 2,304 values.

See `experiments/MAEST_HF_PROTOCOL.md` and
`experiments/MAEST_HF_IMPLEMENTATION_NOTE.md`.

## Integrity checks

- Frozen plan SHA-256:
  `8cbc02cdb76502716ed09dbb00e60fb1df04fd9a4986c0f4c6a89d8e941df6a1`.
- Feature shape: 1,206 tracks x 2,304 values; all values finite.
- Feature SHA-256:
  `cfb693735b4affcd966375926026fb39c45348bb60d19cef8e244ac436aeca52`.
- Development metrics SHA-256:
  `db00a931f5df83f282f4d8b3d5657f622fff5381e505d7724410e1b6c0b47976`.
- Test predictions: 0.
- Single-track pilot: two identical outputs, model in evaluation mode, no
  parameter requiring gradients.
- Unit tests after implementation: 39 passed.

## Selected heads

| Label | C | Class weight | Mean grouped-CV AP |
| --- | ---: | --- | ---: |
| electronic | 0.001 | none | 0.7744 |
| pop | 0.001 | none | 0.5362 |
| ambient | 0.001 | balanced | 0.4715 |
| rock | 0.001 | none | 0.7045 |

## F1-oriented development validation

Training-OOF thresholds were 0.400, 0.275, 0.375, and 0.275.

| Label | Positives | Precision | Recall | F1 | AP | Predictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| electronic | 119 | 0.722 | 0.655 | 0.687 | 0.762 | 108 |
| pop | 49 | 0.348 | 0.490 | 0.407 | 0.339 | 69 |
| ambient | 70 | 0.374 | 0.743 | 0.498 | 0.455 | 139 |
| rock | 50 | 0.649 | 0.740 | 0.692 | 0.735 | 57 |

Micro precision was 0.512, micro recall 0.663, micro-F1 0.578, macro AP 0.573,
and output coverage was 261/303 = 0.861. The four-label usefulness goal failed.

## Precision-target development validation

Training OOF supported thresholds only for `electronic` (0.625) and `rock`
(0.525). The policy suppressed pop and ambient rather than emitting unsupported
high-confidence labels.

| Label | True positive | False positive | Precision | Recall | Predictions |
| --- | ---: | ---: | ---: | ---: | ---: |
| electronic | 44 | 6 | 0.880 | 0.370 | 50 |
| pop | 0 | 0 | undefined / suppressed | 0.000 | 0 |
| ambient | 0 | 0 | undefined / suppressed | 0.000 | 0 |
| rock | 23 | 6 | 0.793 | 0.460 | 29 |

Combined micro precision was 0.848, recall 0.233, and coverage 79/303 = 0.261.
This is useful evidence for a selective electronic/rock classifier, but it does
not satisfy a four-label model or the minimum 0.50 coverage rule.

## Comparison on the same development set

| Metric | MERT-v0 | MERT-v1 | MAEST |
| --- | ---: | ---: | ---: |
| micro precision | 0.437 | 0.507 | **0.512** |
| micro recall | 0.590 | 0.528 | **0.663** |
| micro-F1 | 0.502 | 0.517 | **0.578** |
| macro AP | 0.524 | 0.535 | **0.573** |
| output coverage | **0.881** | 0.776 | 0.861 |

MAEST is the strongest of the three representations overall. Its largest gains
are electronic AP (0.762) and rock AP (0.735). It does not improve pop AP over
MERT-v1, and ambient remains difficult despite higher recall.

## Post-outcome diagnosis

This section is exploratory because the development-validation results were
already known. It cannot be reported as a new unbiased performance estimate.

At validation-ranked prefixes with at least 0.80 precision:

- electronic: 90 predictions, precision 0.800, recall 0.605;
- rock: 25 predictions, precision 0.800, recall 0.400;
- pop: no prefix of at least five predictions reached 0.80;
- ambient: no prefix of at least five predictions reached 0.80.

These prefixes show that electronic and rock have a usable high-precision
region. Their cutoffs were observed after validation and are not silently
substituted for the frozen OOF thresholds.

Learning curves rise consistently for electronic and rock. Pop improves from
25% to 75% of training artists, then changes only slightly. Ambient peaks at
75% and falls slightly at 100%, which is consistent with noisy or ambiguous
targets rather than a simple shortage of same-source tracks.

Under the F1 policy, pop has 45 false positives and ambient has 87. Twelve pop
false positives carry a declared neighbouring tag such as `indie` or `chanson`;
18 ambient false positives carry a neighbour such as `easylistening`,
`experimental`, `lounge`, or `soundscape`. This is a reason to listen, not proof
that source labels are wrong. Exact counts, curves, and cases are stored in
`outputs/maest_hf/post_outcome_diagnostic.json`; 40 fixed cases are listed in
`data/maest_error_listening_review.csv`.

## Human listening review

The project author completed all 40 fixed cases while seeing the queried label
but not the model score, source tags, or error type. Fourteen of 20 apparent
false positives audibly contained the queried label, while 11 of 20 apparent
false negatives did not audibly contain the source label in the first 30
seconds. Three false positives and four false negatives were more consistent
with genuine model errors; eight cases remained uncertain.

This deliberately selected error sample cannot produce a corrected accuracy
estimate. It does, however, provide strong qualitative evidence that incomplete
track-level tags and first-30-second mismatch materially affect the measured
errors. See the full procedure, per-label table, reasons, and limitations in
[the listening review](MAEST_LISTENING_REVIEW.md).

## Decision

This development result was not promoted as an 80%-precision four-label
classifier. After the listening review, a separate final protocol was frozen
before the holdout was opened.

The fixed listening review is now complete. Do not silently relabel development
data, recalculate headline metrics, add more same-source data, or try another
encoder. Freeze a final policy before opening the holdout. The recommended
portfolio direction is a four-label research prototype that explicitly studies
the gap between source tags, 30-second perceptual labels, and model predictions;
the electronic/rock strict policy can remain a secondary selective-use result.

The subsequent one-time 239-track evaluation is reported separately in
[FINAL_RESULTS.md](FINAL_RESULTS.md). It did not change any development choice.
