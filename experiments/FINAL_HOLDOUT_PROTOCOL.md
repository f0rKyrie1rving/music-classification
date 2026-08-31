# Final MAEST holdout protocol

Frozen after all development outcomes and the 40-case listening review were
known, but before any new-holdout audio was downloaded, encoded, or scored.

## Purpose and boundary

Evaluate one final, already selected Discogs-MAEST broad-genre model on the 239
`new_holdout` tracks frozen in `data/expanded_manifest.json`. The 90 historical
test tracks remain excluded because their earlier MFCC outcomes were observed.
No holdout result may change the representation, labels, classifier settings,
thresholds, preprocessing, or reported evaluation policy.

The holdout has 89 artists and no artist overlap with the 903 training or 303
development-validation tracks. The source-label positives known from the
preexisting manifest are electronic 70, pop 50, ambient 48, and rock 36; 78
tracks have none of the four broad targets.

## Fixed final model

- Encoder: pinned official `mtg-upf/discogs-maest-30s-pw-129e-519l` revision
  already audited in `experiments/MAEST_HF_IMPLEMENTATION_NOTE.md`.
- Input: exactly the first 30 source seconds, mono, resampled with soxr HQ to
  exactly 480,000 samples at 16 kHz.
- Representation: seventh transformer block (`hidden_states[7]`), concatenating
  CLS, DIST, and the mean of remaining signal tokens into 2,304 values.
- Heads: one StandardScaler plus fixed LogisticRegression configuration per
  label. Configurations are copied exactly from the completed MAEST development
  report; there is no new search or cross-validation.
- Final fitting rows: all 1,206 development tracks (903 train plus 303
  development validation). The representation, configurations, and thresholds
  were chosen before this refit. No holdout artist occurs in these rows.
- Primary policy: the already selected F1 thresholds electronic 0.400, pop
  0.275, ambient 0.375, and rock 0.275.
- Secondary selective policy: electronic 0.625 and rock 0.525; pop and ambient
  remain suppressed with threshold 1.01. Suppression is reported as zero recall,
  not as successful high precision.

## Acquisition and integrity

Only rows whose frozen `phase_role` is `new_holdout` may be downloaded. Byte
ranges, decoded WAV hashes, format, finite values, and RMS are recorded. The
same predeclared acquisition exception used for development is allowed only
when a checksum-verified complete MP3 is no more than 0.1 seconds short; only
the missing tail may be zero-padded. Tracks cannot be replaced.

MAEST feature extraction is resumable but must cover all 239 IDs in frozen
order. The cache must be finite with shape 239 x 2,304 and must record the plan,
audio, source-code, runtime, and model hashes. Historical-test audio is never
encoded by the final extractor.

## One-time source-label evaluation

Fit the four final heads once on all 1,206 development representations, predict
all 239 holdout tracks once, and save every score and decision. Report for both
policies:

- per-label positives, predictions, TP, FP, FN, precision, recall, F1, and AP;
- micro precision, recall, and F1;
- macro precision, recall, F1, and AP;
- output coverage and tracks with output;
- fixed-seed track bootstrap 95% percentile intervals using 2,000 replicates.

The provisional four-label goal is met only if the primary policy has micro
precision >= 0.80, coverage >= 0.50, and every label has precision >= 0.80,
recall >= 0.30, and at least 10 predictions. Results are reported regardless of
outcome. The holdout is never reused as development data.

## Independent perceptual check

Before model scores are produced, freeze 40 unique holdout queries: for each
label, five source-positive and five source-negative tracks selected with seed
20260830. The reviewer later sees only the queried label and audio, not the
source target, model score, model decision, or error type. Answers are `yes`,
`no`, or `uncertain` with a short reason.

This small single-reviewer check is secondary qualitative evidence. It cannot
replace the source-label holdout metrics, establish population accuracy, or be
used to tune the evaluated model. Agreement with source targets and model
decisions will be reported only after the answers are complete.

## Disclosure

MAEST pretraining track-level overlap with MTG-Jamendo cannot be audited from
public data, so the result is artist-disjoint within this project but not proven
pretraining-disjoint. Source tags are incomplete whole-track annotations, while
the system evaluates only the first 30 seconds. Model weights stay local and
are not redistributed while their publication terms remain unresolved.
