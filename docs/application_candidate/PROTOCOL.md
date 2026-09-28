# Fixed candidate construction, 2026-09-27

The user authorized training a fixed candidate after the exploratory factorial
study. This is a NEW construction protocol, recorded after those outcomes were
known, before this procedure's fits and threshold selection. No historical
experiment is retroactively changed. There is no claim of fresh validation here.

## Data and fixed architecture

Use exactly the 1,915 / 908 tracks / artist IDs from the audited factorial pool,
in the same order and with unchanged source-tag proxy targets and MAEST features.
Keep the retirement ledger and complete 2,195-track / 1,008-artist exclusion union.
No new audio, pseudo labels, label editing or listening answers enter training.

Only pop and ambient are newly fitted: standardized logistic regression,
C=0.0003, no class weights, float64, the already specified solver settings.
Scaler and class counts are fitted only to each actual training subset.
Fit the final two heads once on all 1,915 rows after threshold selection.

Copy electronic and rock parameter VALUES, intercepts and both policy thresholds
from the actual existing application artifact. Preserve their exact float64
inference arithmetic. Lossless promotion of old stored float32 values to float64
is allowed; do not round newly fitted float64 parameters. Thus this is a mixed
candidate, deliberately avoiding retraining previously unaffected app labels.
It differs from the earlier refitted-recipe comparison and requires its own
future comparison against the true installed baseline.

## Final threshold selection

Make one label-free cohort/whole-artist-balanced FOUR-fold assignment on all
development rows with the fixed config seed. No seed search or repeated attempts
are allowed. Each row receives exactly one out-of-fold prediction, from a model
that has not trained on any song by its artist ID.

For each fold fit four focus heads: old pop C=.001 unweighted; old ambient C=.001
balanced with its fit-only log(n_positive/n_negative) offset; and the two new
unweighted C=.0003 heads. These 16 fits support threshold selection only.
Every model has its own fitted scaler and prior. Old decisions use the original
application raw f1 cutoff exactly. The original-rule candidate for new heads
uses that baseline cutoff mapped to the corresponding fit's probability scale.

For each new focus head, use the same previously fixed threshold selector:
its row-specific original policy and constants .025 through .975, step .025.
Both use the baseline-original OOF decisions as reference. Require FP not greater,
recall loss at most .03 and F1 not smaller, with 1e-12 metric tolerance. Minimize
FP, then maximize TP, then prefer original, then higher constant threshold.
If none is feasible, keep THIS new head's original rule and report infeasibility;
do not swap models, change the seed or invent another threshold grid.

A selected constant becomes the final cutoff unchanged. For selected original
rules, map the old raw threshold using the baseline's FULL-development prior
(zero for pop; log(399/1516) for ambient). No full baseline head is required to
compute this offset. This construction uses exactly 18 newly fitted heads:
16 OOF support heads and two final heads, not a new model search.

The OOF labels are deliberately used to choose thresholds, so their selection
scores are development diagnostics, NOT an independent performance estimate.
Do not apply the prior outer evaluation gate to these newly selected OOF scores
or claim that the final artifact inherits its 10.18% result.

## Bundle and inference

Save a separate candidate under artifacts/candidates/20260927_v1. Do not modify
the application's original weights, metadata, correction, version or entrypoint.
The candidate has its own strict loader and uses non-executable NumPy arrays.
All candidate heads are unweighted, so all probability offsets are exactly zero;
never apply the old ambient weight correction to the new unweighted head.

Decisions use unrounded probabilities. To preserve electronic/rock controls,
compute each row using the actual app's sum of standardized-feature products and
exp(-logaddexp(0,-logit)), not matrix-multiplication inference with different
roundoff. Both copied labels retain exact app scores and decisions. The optional
precision_target policy keeps pop/ambient disabled at 1.01 and the copied labels'
original thresholds, so that policy's selected tags are wholly unchanged. Its
pop/ambient displayed scores still change with the new heads.

Include the training receipt, selected-threshold audit, freeze receipt and future
exclusions, each bound by hash in metadata. The freeze preserves source snapshots,
input and baseline artifact hashes, runtime, all training IDs, folds and rules
before any fit. Every fit has started/complete/failed receipts; failed or partial
fits may not be silently repeated. Training runs are locked against concurrency.

## Construction verification and stopping point

Verify the four-fold boundaries and exact OOF coverage; independently replay
saved heads and selector arithmetic; verify fit-only scales, means and priors;
check final focus counts, class weights and C; check parameter export and loading.
Compare copied labels with actual predict_app on all 1,915 cached vectors and
synthetic boundaries, and check both policies. Extract one existing local 30-second
control audio with the pinned encoder, compare against its cached float32 vector
(absolute tolerance 1e-5), and run the candidate and actual app on the same vector.
The audio smoke check is a functioning check, never new performance evidence.

No training-set accuracy or fitted-score improvement is claimed as validation.
Keep all historical source/output files and the installed app unchanged. This
stage ends with a runnable fixed candidate and a verification report, without
new-song collection, publication, installer replacement or a release version bump.
Before any later new-song acquisition, freeze a separate sample, baseline and
acceptance protocol; keep this candidate's hashes unchanged throughout that test.
