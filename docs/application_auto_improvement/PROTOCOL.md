# Automatic application improvement: development gate

Fixed locally before any new fitting or candidate scores in this experiment.
Previous project results, including the 204-song application check and its error
audit, are known. This is exploratory development, not an untouched confirmatory
study. Source tags remain the reference labels; no human answers are required or
inferred. Neither source labels nor previous results will be rewritten.

## Data boundary

Use only the original 1,206 development songs / 469 artist IDs from
`experiments/final_holdout_plan.json:fit_ids`. Align the verified frozen
`outputs/maest_hf/features.npy` (1,206 x 2,304 float32) and its metadata. Check
the audio-manifest and encoder/extractor provenance hashes. Do not load feature
values or predictions from the 239 historical holdout, earlier 266-song
validation, or recent 204-song check for fitting, selection or this gate.
Their known outcomes motivate the question but do not enter the optimizer.

## Question and bounded candidates

Try to reduce the pop/ambient false-positive count without materially lowering
recall or worsening probability error. Only these two classifier heads and their
decision thresholds may change. Electronic and rock are exact baseline controls.
No new audio encoder or feature extraction is needed in development.
This workflow optimizes and gates only the default `f1` policy. The separate
historical `precision_target` policy is outside this experiment's scope.

Within every training subset, reconstruct the application baseline recipe:
StandardScaler fitted only on that subset, independent logistic heads with
C=0.001, ambient class_weight=balanced and other labels unweighted. Use the
fixed historical raw F1 thresholds from the final plan. Ambient display
probability receives log(n_positive/n_negative) using ONLY that model's training
subset. Other probabilities remain raw. Preserve raw probabilities separately
for baseline decisions; never apply the old raw threshold to corrected scores.

Candidates are unweighted pop/ambient logistic heads with C in
{0.0001, 0.0003, 0.001}. The baseline recipe itself is also eligible, allowing a
threshold-only change. C is the inverse regularization strength: the smaller
values constrain the head more strongly. Do not expand this grid after results.

## Nested artist separation and deterministic selection

Use 5 outer folds and 4 inner folds. Assign artists deterministically by a seeded
hash and balance track counts without reading targets. Each artist stays in one
fold. Each row receives one outer prediction whose fitting and NEW candidate /
threshold choices in this run exclude that row and all its artist's songs.
The fixed historical baseline representation, C and original thresholds were
previously chosen using this development pool; nested splitting cannot undo
that earlier exposure. This is not a wholly unbiased first-use nested test.
Standardization and class weights
are independently fitted inside every training subset. A single-class training
subset or fitting failure stops the run; do not redraw the split.

For each outer training pool, collect complete inner out-of-fold baseline and
candidate predictions. The selector receives only these inner outcomes:

1. Reject candidate heads with Brier or log loss above baseline (tolerance 1e-12).
   Choose the lowest-Brier remaining head, including baseline. Numerical ties
   prefer baseline, then the smaller C. The comparison concerns probability
   outputs, not threshold decisions.
2. With that head fixed, evaluate the frozen 0.025..0.975 threshold grid plus the
   exact original raw threshold. Use raw decision scores for a baseline head;
   use unweighted probabilities for an unweighted head. Require recall at least
   baseline recall minus 0.03, F1 at least baseline F1 and FP no more than baseline.
   Among feasible thresholds minimize FP, then maximize TP, then choose the
   higher threshold. If none is feasible, use the complete baseline head,
   corrected probabilities and original raw decision rule.
3. Refit the selected recipe only on the outer training pool, then score the
   outer fold once. Retain every inner prediction, selection record, fitted
   model, outer prediction and fold count.

Threshold selection can reduce positive calls and therefore increase missed
labels. Report TP, FP, FN, precision, recall and F1 jointly, by label and overall.
Also report macro binary Brier and log loss. These are relative to source labels,
not human-adjudicated musical truth.

## Development pass/fail fixed before fitting

Pool the 1,206 outer predictions once (not the five folds as independent studies).
Proceed to a fresh-song confirmation only if ALL hold:

- pop+ambient FP is at least 10% lower than baseline;
- neither pop nor ambient individually increases its FP count;
- micro recall loses at most 0.03 absolute; each label loses at most 0.05;
- micro F1 does not decrease;
- macro Brier and macro binary log loss do not increase;
- electronic/rock probabilities and decisions are exactly unchanged, no rows
  are missing and all fit/selection/test artist boundaries are disjoint.

These margins are explicit application tradeoffs, not statistical significance
or a guarantee that the final model improves. Report each fold as a diagnostic;
do not pick the best fold or treat the five folds as independent datasets.

If the gate fails, retain the negative result, stop this candidate workflow, do
not acquire a new test cohort to try to rescue it, and keep the application as is.
If it passes, repeat the SAME 4-fold inner selection on all 1,206 development
songs, refit on all of them, and freeze a candidate artifact. The outer fold
choices are not votes for final hyperparameters. Freeze a separate sampling and
acceptance protocol before any new-song acquisition or scores. A passed
development gate alone never authorizes calling the candidate validated or
replacing the current default application.

At final candidate assembly, any selected baseline head (including the mandatory
electronic/rock controls) uses the original packaged parameters and correction
from `artifacts/final_heads.npy`, `final_heads.json` and `score_correction.json`.
Keep the full-data baseline refit separately for audit. This avoids small
float32/float64 refitting differences becoming accidental application changes.
The nested development comparison still uses training-fold-only refits of both
recipes. Fresh-song inference must preserve the original arithmetic for copied
baseline heads; their parameters, probabilities and original unchanged control
thresholds must match the existing application.

Future new-song confirmation must conservatively exclude all documented prior
candidate tracks and artists, currently 2,195 IDs / 1,008 artists, including all
266 candidates from the 08–11 frame and its 28 DNS failures. Metadata-only
feasibility checks of potential later shards are logged separately. Newness is
relative to recorded project IDs, not an independent dataset; artist aliases,
encoder pretraining overlap and missing source labels remain limitations.

## Method references

The separation of tuning and evaluation follows the principle in the official
[nested cross-validation example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
Artist grouping follows the grouped-data guidance in the official
[cross-validation documentation](https://sklearn.org/stable/modules/cross_validation.html).
The exact algorithm, margins and grid above are project design choices, not
claims that those sources establish them as optimal.
