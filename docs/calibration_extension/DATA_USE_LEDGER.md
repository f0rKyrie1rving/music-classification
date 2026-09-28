# Data use after the mechanism/transfer/refitting extension

This extension introduces no new song IDs. It refits models using the already
observed 1,206 development songs / 469 artists, with different grouped splits.
It then scores the already observed 266-song / 200-artist cohort with additional
fitted MAEST, MERT-v0 and MERT-v1 pipelines. No member of that cohort is now fresh
evidence for further model/calibrator selection.

The earlier 1,535-song / 583-artist project history and the additionally inspected
394-song / 254-artist candidate frame remain recorded in the
[previous ledger](../calibration_validation/DATA_USE_LEDGER.md). For future untouched
evaluation, retain the conservative exclusion of all 837 artist IDs across those
two sources. This is conservative retirement of inspected metadata, not a claim
that every candidate song was fitted or scored.

No 239-song historical application holdout prediction is newly computed in this
extension. Existing dataset roles, manifests and earlier reports are not rewritten.
Project-external data exposure, artist aliases and encoder pretraining overlap
remain unverified. A different random split of observed songs is not a new dataset.
