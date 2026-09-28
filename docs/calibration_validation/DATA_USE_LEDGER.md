# Data-use ledger after validation

This run has now exposed the model outcomes for 266 songs from 200 artists. Its IDs
must not be presented as untouched evidence for subsequent changes to the model,
calibrator, selection rule, ontology or evaluation procedure.

- Earlier project history: 1,535 tracks / 583 artists, recorded in
  `data/expanded_manifest.json` (SHA-256
  `01d93dbe0474e517861e0d0cbe95ca4a4e8dc769f180ec74d2e3d34490689077`).
- This candidate frame: 394 tracks / 254 additional artists in
  `outputs/calibration_validation/20260927_v1/candidate_pool.json`. Its metadata was
  inspected, but only the selected sample was downloaded and predicted.
- This selected and scored sample: 266 tracks / 200 artists in
  [selected manifest](results/selected_manifest.json) and
  [observed manifest](results/manifest.json). There were no failures/replacements.
- At minimum, future fresh tests must exclude all previously scored or fitted artists,
  including these new 200. A conservative future exclusion can additionally retire
  all 254 artists in this inspected candidate frame, for 837 artists together with
  the previous 583. This is a planning choice, not a claim that all 394 candidates
  were predicted or used for fitting.
- The external-use-history question remains unanswered; see
  [the recorded exposure status](results/exposure_history.json).

Do not edit old manifests to conceal this additional history. Combine explicit
historical and new-use records in any future sampling protocol. Previously recorded
encoder-pretraining and artist-alias uncertainties remain unresolved.
