# Final Discogs-MAEST holdout results

Final run: 2026-08-30. The model, thresholds, holdout IDs, and reporting rules
were frozen before final MAEST holdout scoring. No holdout result was
used to change the evaluated system.

## Evaluation boundary

- Final fit: all 1,206 development tracks (903 train and 303 development
  validation), comprising 469 artists.
- Final holdout: 239 tracks from 89 different
  artists, with no artist overlap with the fit rows.
- Historical test: 90 previously observed tracks, excluded from final fitting,
  feature extraction, and prediction.
- Targets: the same frozen electronic, pop, ambient, and rock broad-label
  ontology used throughout expanded development.
- Representation: fixed 2,304-value Discogs-MAEST block-7 CLS, DIST, and mean
  signal-token vector.
- Heads and thresholds: copied exactly from development; no holdout tuning.

Review addendum (2026-09-12): 40 of these artists also appeared in the previously
observed historical test, accounting for 150 holdout tracks. `track_0543400` was
already downloaded and processed as a learning/decoding sample. It was excluded
from the original 480-track subset but reintroduced during expansion. No holdout
artist occurs in the final fit; this does not establish that all holdout tracks
and artists were previously unobserved. The original scores and all 239 rows
are retained. See [post-hoc exposure and uncertainty checks](EVALUATION_REVIEW.md).

The frozen protocol is in
[`experiments/FINAL_HOLDOUT_PROTOCOL.md`](experiments/FINAL_HOLDOUT_PROTOCOL.md).

## Primary four-label policy

Thresholds were electronic 0.400, pop 0.275, ambient 0.375, and rock 0.275.

| Label | Source positives | TP | FP | FN | Precision | Recall | F1 | AP | Predictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| electronic | 70 | 56 | 38 | 14 | 0.596 | 0.800 | 0.683 | 0.749 | 94 |
| pop | 50 | 31 | 31 | 19 | 0.500 | 0.620 | 0.554 | 0.506 | 62 |
| ambient | 48 | 40 | 64 | 8 | 0.385 | 0.833 | 0.526 | 0.444 | 104 |
| rock | 36 | 27 | 22 | 9 | 0.551 | 0.750 | 0.635 | 0.676 | 49 |

Micro precision was **0.498**, micro recall **0.755**, micro-F1 **0.600**,
macro-F1 **0.600**, macro AP **0.594**, and output coverage was 206/239 =
**0.862**. The predeclared 80% four-label goal was not met.

Track-bootstrap 95% intervals using the frozen 2,000 replicates were:

| Metric | Estimate | 95% interval |
| --- | ---: | ---: |
| micro precision | 0.498 | 0.444–0.553 |
| micro recall | 0.755 | 0.695–0.818 |
| micro-F1 | 0.600 | 0.549–0.651 |
| macro AP | 0.594 | 0.538–0.671 |
| output coverage | 0.862 | 0.820–0.904 |

These intervals describe holdout sampling variability. They do not account for
source-label incompleteness, reviewer subjectivity, or unknown MAEST pretraining
overlap. They also treat tracks as independent sampling units. The earlier
expansion protocol promised artist-grouped uncertainty, while the final protocol
specified track bootstrap before scoring. This protocol change and supplementary
artist-cluster intervals are documented in [EVALUATION_REVIEW.md](EVALUATION_REVIEW.md).

## Secondary electronic/rock selective policy

The development-frozen thresholds were electronic 0.625 and rock 0.525; pop
and ambient were suppressed at 1.01.

| Label | TP | FP | FN | Precision | Recall | Predictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| electronic | 40 | 20 | 30 | 0.667 | 0.571 | 60 |
| pop | 0 | 0 | 50 | suppressed | 0.000 | 0 |
| ambient | 0 | 0 | 48 | suppressed | 0.000 | 0 |
| rock | 19 | 8 | 17 | 0.704 | 0.528 | 27 |

Combined micro precision was **0.678**, recall **0.289**, micro-F1 **0.405**,
and coverage was 87/239 = **0.364**. Its micro-precision 95% interval was
0.577–0.772. The 0.848 development precision did not transfer to the holdout,
so this policy must not be presented as an 80%-precision mode.

## Development-to-holdout comparison

| Primary-policy metric | Development validation | Final holdout |
| --- | ---: | ---: |
| micro precision | 0.512 | 0.498 |
| micro recall | 0.663 | 0.755 |
| micro-F1 | 0.578 | 0.600 |
| macro AP | 0.573 | 0.594 |
| output coverage | 0.861 | 0.862 |

The ranking quality, F1, and coverage were stable on artists not used during
development. This is useful generalization evidence, even though the requested
precision target was not achieved. Ambient remains the main source-label
precision problem; pop ranking was better on holdout than on development.

## Integrity and artifacts

- Final plan SHA-256:
  `00e902fef9a9a1d49b5fabf2be5d17cc54f6cf11633e369dd20db239c22e8be4`.
- Holdout audio audit SHA-256:
  `a84de96f37abec74a478c9a1d95dda99948a9260ed16c62d0e4ae37fdad2ad38`.
- Holdout feature SHA-256:
  `39217b8c7fd148405d813f7b4249eed3f7099cf2194430a452efa50632b74211`.
- Metrics SHA-256:
  `f9328df119a074f936f4450ab023e2e7de384d769dd51a5648fb223dacad6372`.
- Historical-test predictions: 0; new-holdout predictions: 239.
- All 239 WAV files passed receipt hash, exact-format, finite-value, and RMS
  checks; no short-source repair was needed.

Generated local artifacts include `holdout_predictions.csv`,
`holdout_scores.npz`, and `final_broad_genre_model.joblib` under
`outputs/final_maest/`. Audio and feature caches remain local. A curated copy of
all targets, scores, decisions and original metrics is now available in
[`data/evaluation/`](data/evaluation/README.md), with `evaluate_release.py` for
independent recalculation. The original artifact digests above remain historical.

## Interpretation and decision

This is a valid four-label research prototype and a useful genre-ranking aid,
but not a reliable 80%-precision classifier. The primary policy is retained as
the final research result because it covers most tracks and preserves all four
labels. The strict policy remains a secondary comparison, not the user-facing
default.

The earlier targeted listening review found substantial whole-track-label and
first-30-second mismatch, but it cannot correct these source-label holdout
metrics. The separately frozen 40-query perceptual check is now complete: 36
answers were decisive, source targets agreed with the reviewer on 25/36
(69.4%), and final model decisions agreed on 29/36 (80.6%). Against the
reviewer's decisive answers, descriptive model precision was 13/17 = 76.5%.
These balanced, single-reviewer counts are qualitative evidence rather than a
replacement accuracy estimate, and they were not used for tuning. See the
[perceptual-check report](BLIND_REVIEW_RESULTS.md). The reviewer interface hid
source targets and model outputs, but question order could reveal source-target
strata; this was an order-structured single-reviewer audit.
