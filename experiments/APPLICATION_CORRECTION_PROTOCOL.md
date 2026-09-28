# Application v1.1 score correction

The fixed formula is chosen from the preceding research for the actual released
1,206-track classifier. No new classifier or fitted calibrator is selected here.
The original `final_heads` and `predict_maest.py` remain the v1.0 baseline.
The executable evaluation is frozen after unit testing and an independent numerical
precheck. The cohorts and the broader correction idea have already been examined;
this is an engineering validation, not a preregistered or blinded experiment.

Apply the inverse balanced-class-weight log-odds offset only to heads trained with
`class_weight='balanced'`. Count positives and negatives using only the exact
`fit_ids` in the original plan and its frozen ontology/manifest. For ambient these
are 244 positives / 962 negatives; other heads have zero offset. No held-out label
or research-model calibration parameter is used to set this correction.

Map active thresholds through the same monotone function. Preserve disabled >1
threshold sentinels. Compute tag decisions with the original unrounded raw score
and raw threshold, making classification, recall, false positives and within-label
rankings unchanged by construction. Do not claim increased recognition accuracy.
Scores across different labels may change relative order.

Validate the actual release on two separately reported, already observed cohorts:
the 239-track historical holdout (89 artist IDs) and the previously used 266-track
cohort (200 artist IDs). Reconstruct historical raw scores before comparing. Allow
the recorded float32 score-rounding error, but require the new raw path to exactly
match the original predictor's returned fields. Both are retrospective checks, not
new independent tests. They exclude classifier-fit artist IDs.

Integration gate: macro Brier AND macro binary log loss must be lower on each
cohort; all selected labels under both existing policies must match v1.0. Report
per-label metrics and paired artist-cluster bootstrap intervals conditional on
the fixed model and correction (2,000 draws, seeds 2026092711/2026092712). Include
all tracks, with no replacement, outcome filtering or parameter adjustment.

Save a separate correction artifact bound to the original weights, metadata, plan,
fit-ID sequence and manifest hashes. The production loader verifies the exact
model binding and formula; evaluation independently reconstructs the fit counts.
Expose raw-score mode for comparison. The correction is an estimate based on the
project training distribution, not a guarantee for all user music. Other labels
are not newly calibrated. New data would strengthen future generalization claims
but is not claimed by this reversible local application update.

The fixed weighting correction is established methodology; see
[Caplin, Martin and Marx (2022)](https://arxiv.org/abs/2205.04613).
