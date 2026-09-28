# Frozen exploratory calibration protocol, version 1

This protocol is fixed before generating any new evaluation predictions. The historical
239-track results were already public and described in the handoff. This is a local
execution record, not an externally registered preregistration.

## Inputs and estimand

Use the verified original 1206 x 2304 float32 MAEST block-7 features, original ID order,
and frozen broad targets (electronic, pop, ambient, rock). The encoder is fixed; features
are not refitted. Study track-weighted probability quality relative to observed tags in
the historical enriched sample. Both multi-positive and all-zero targets are retained.
The 239-track holdout is a historical/post-hoc descriptive audit only; the older 90 tracks
are excluded. Neither supplies calibration observations for the new experiment.

## Split rule fixed before outcomes

Five seeds are enumerated in config.json. The first is primary. For each seed, permute
sorted unique artists 256 times, allocating round(60%) artists to fit, round(20%) to
calibration, and the remainder to evaluation. All tracks of an artist stay together.
Candidates must have both outcomes and at least three positive artists for every label
in each subset. Select the earliest minimum of: sum of squared track-fraction deviations
from [0.6,0.2,0.2], plus mean squared subset-prevalence deviations from pool prevalence,
plus mean squared positive-artist allocation deviations from the same target fractions.
Only labels and artist IDs enter this rule. Log every candidate objective, selected
indices, IDs, artists and counts. Abort if no candidate qualifies; do not relax rules.
This label-balanced evaluation sampling is itself a limitation on population inference.

## Fitting

Fit each StandardScaler and logistic head on classifier-fit rows only. Reuse C=0.001,
ambient balanced weights and other labels unweighted; lbfgs, max_iter=3000, tol=1e-4,
random_state=2026. No hyperparameter search. Save scaler and head numerical parameters.
Compute full float64 decision_function logits on calibration and evaluation rows.

Fit independently for each label on calibration logits/targets only:

- identity: sigmoid(z);
- monotone Platt-type sigmoid: sigmoid(a*z+b), a in [0.001,100], b in [-20,20];
- binary temperature: sigmoid(a*z), a=1/T in [0.001,100], hence T in [0.01,1000].

Optimize mean unweighted binary log loss by L-BFGS-B with analytic gradients, starting
at identity. No label smoothing, regularization, class balancing or resampling in
calibrator fitting. This is a constrained Platt-type model, not an exact implementation
of Platt's original smoothed-target estimator. Bounds prevent reversed ranking and
unbounded estimates in sparse data. Record bounds reached, parameters, optimizer status
and objective. Save failures and abort a failed run; never replace a failed method or
silently drop it. No softmax or normalization across labels. Isotonic and beta are not
included in this first protocol. Evaluation labels enter scoring only after fitting.

## Outcomes and intervals

Primary: mean of four binary Brier scores, equally weighted by label, lower is better.
Auxiliary: per-label Brier and binary log loss (clip probabilities to [1e-12,1-1e-12]),
reliability curves, score histograms, and positive-label ECE with fixed equal-width
5 bins (primary visualization) and 10 bins (sensitivity). Bins are left-closed,
right-open except the final bin includes 1. Empty bins are explicitly recorded and
omitted from curves. AP is descriptive and undefined single-class cases are null.
No threshold/F1 optimization or classification-improvement claims.

For the primary split only, use 2000 artist-cluster bootstrap draws with a fixed seed.
Draw evaluation artists uniformly with replacement and include every track in each
sampled cluster. Preserve all four targets and all methods for each sampled track.
Compute paired calibrated-minus-identity Brier and log-loss differences; negative favors
calibration. Report 2.5/97.5 percentile intervals for macro and label metrics. These
pointwise intervals condition on the fitted model/calibrator and omit training and
calibration sampling uncertainty; there is no multiple-comparison adjustment.
Show all five split results as stability descriptions. Do not treat overlapping splits
as independent samples, pool repeated predictions, or apply an ordinary t-test.

## Historical audit and interpretation

Stage A recomputes counts, mean scores, Brier/log loss, five- and ten-bin reliability
curves and distributions from saved historical probabilities; it fits no calibrator.
No inferential intervals are added to this descriptive stage. Do not choose methods or
splits from its plots. Brier and log loss measure overall probability quality, not pure
calibration in isolation. Inspect reliability and distributions alongside scores.

The pool previously informed model selection, so even artist-separated results remain
exploratory. Target enrichment, incomplete/noisy labels, first-30-second/whole-track
mismatch and unknown MAEST/Jamendo pretraining overlap constrain generalization.
Any stronger confirmation requires new, previously unused songs AND artists plus a
separately fixed sampling plan. Null and unfavorable outcomes must remain in the report.

## Reproducibility

Save this protocol/config and their hashes, all split candidates, source snapshots and
hashes, input hashes, base Git commit/status, dependencies, timestamps, full-precision
predictions, fit/calibration IDs, parameters, metrics and bootstrap draws. Commands
refuse to overwrite existing freezes or result directories. Original release files
and historical locks are read-only inputs. Reuse the verified .venv-improve interpreter;
only numpy/scipy/scikit-learn/matplotlib are imported from third-party packages.
