# Data expansion protocol — frozen before new audio acquisition

## Purpose

The first model showed that a frozen music representation helps, but the reused
90-track development validation set still has low precision.  The next phase
tests whether broader data and a cleaner evaluation improve a four-label model:
`electronic`, `pop`, `ambient`, and `rock`.

## Fixed data roles

- Source pool: every eligible non-ND track already recorded in
  `data/source/training_pool.json` from MTG-Jamendo archive shards 00–03.
- Training: all eligible official `train` rows in that pool.
- Development validation: all eligible official `validation` rows.  These rows
  may be used for model selection and threshold choice; results are not final.
- Historical test: the 90 test tracks in the original frozen manifest.  Their
  results were already observed, so they are excluded from the new final score.
- New holdout: every other eligible official `test` row in the pool.  Its track
  IDs are frozen before the expanded model is fitted or scored.

The official split keeps training artists separate from validation and test
artists.  Some new-holdout artists may also occur in the historical test; this
is documented, and uncertainty will be grouped by artist.  No test audio is
used to choose the representation, classifier, hyperparameters, or thresholds.

## Acquisition and integrity

- Extend only the four existing TAR indexes; do not silently add another shard
  or change eligibility rules.
- Download byte ranges, decode the first 30 seconds, convert to mono 22,050 Hz
  PCM-16 WAV, and retain per-track acquisition receipts and WAV SHA-256 hashes.
- Preserve the original 480-track manifest and audio files unchanged.
- Freeze a separate expansion manifest before downloading new audio.

## Error audit

The validation error audit is exploratory because its outcomes were already
known.  Source tags and a declared neighbouring-subgenre vocabulary can identify
cases for listening, but cannot automatically convert a false positive into a
correct prediction.  Any manual relabelling must be stored separately with an
auditor decision and must never be inferred from the model score.

## Model-development boundary

All candidate decisions use training folds grouped by artist plus development
validation.  Candidate models must be fixed before the new holdout is scored.
The holdout is evaluated once for the portfolio report.  The target remains at
least 0.80 precision, but results will also report recall, F1, output coverage,
per-label counts, and artist-grouped uncertainty; no metric is guaranteed.
