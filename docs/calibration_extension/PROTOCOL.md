# Mechanism, representation transfer and refitting variability

This is an exploratory extension motivated by previously inspected results. Freeze
this protocol, configuration, source, inputs and all split IDs before fitting any
new heads. It is not a preregistration or a new independent external confirmation.

## Questions and fixed comparisons

1. Does the ambient class-weighting choice explain part of its calibration gain?
   Fit each representation twice on identical rows: ambient `balanced` versus no
   weights; electronic/pop/rock remain unweighted. Compare ambient raw Brier,
   calibration gains, and their paired difference of differences. Unchanged labels
   are exact negative controls. A weight intervention changes the fitted model;
   this experiment does not prove a universal causal explanation of calibration.
2. Does behavior transfer from frozen Discogs-MAEST to frozen MERT-v0 and MERT-v1?
   Use all 1,206 development tracks and the same splits for all three representations.
   MERT features are the float32 mean of cached transformer layers 1 through 12,
   followed by float64 classifier input; no layer search. All models use standardized
   logistic heads, fixed C=0.001, and identical optimization settings. This is a
   controlled fixed-configuration comparison, not best-tuned model performance.
3. How variable are the results when the fitted pipeline changes? Use the 20 seeds
   in config, each with an approximately 60/20/20 artist-level split. For every seed
   refit scalers and heads on fit rows, calibrators and blend selection on calibration
   rows, and score evaluation rows once. Encoders stay frozen. Report variability
   due to partition/refitting, not uncertainty from pretraining or encoder retraining.

The split procedure is the previous fixed grouped label-balance rule (256 candidate
partitions, selection by labels and sizes only). Never select splits using scores.
Each split uses every development track exactly once and has disjoint artist IDs.
Outer splits reuse songs and overlap; 20 is not the number of independent datasets.

## Seven probability treatments

- raw: expit of fitted head logits.
- prior_offset: add log(n_positive/n_negative) to the balanced ambient logit, using
  fit labels only. For other labels and the unweighted condition this is exactly raw.
  This removes the idealized balanced-weight log-odds offset; with regularization
  and finite data it is not an exact reconstruction of the unweighted classifier.
- intercept: slope fixed at one; fit an intercept in [-20,20] by unweighted
  calibration log loss. This isolates a simple shift from changing slope.
- sigmoid and temperature: previous monotone bounded log-loss fits, unchanged.
- conservative: previous five-fold artist-only OOF blend rule, with lambda in
  {0,.25,.5,.75,1}; select the smallest strength within one paired cluster-SE proxy
  of the OOF minimum. This is a heuristic, not a confidence guarantee.
- blend_min: same OOF predictions and full sigmoid fit, choosing the smallest lambda
  attaining the OOF minimum within 1e-12, without the SE tolerance. Same original
  raw fallback if an inner fit fails. This isolates the effect of the SE rule.

Full calibrators use no evaluation targets. No resampling or silent replacement on
failures. Stop and retain any failed head/calibrator run; any corrective new version
must be documented. Inner policy fallbacks follow the existing declared rule and
are counted. Save optimizer statuses, head parameters, logits, OOF predictions,
probabilities and per-label metrics for numerical reconstruction.

## Estimands and reporting

Primary: track-weighted mean of four binary Brier scores. Secondary: per-label Brier,
binary log loss, AP and ambient mean-score minus observed positive fraction.
The seven treatments are independent binary probabilities; do not normalize them.
No decision threshold, F1 or accuracy improvement is claimed.

Report every seed and condition; summaries include mean, median, quartiles, range,
number of improvements/deteriorations (tolerance 1e-12). For the mechanism, report
ambient (balanced raw minus unweighted raw), each treatment minus its own raw, and
the difference of loss deltas:
`(L_balanced,treatment - L_balanced,raw) - (L_unweighted,treatment - L_unweighted,raw)`.
A negative difference of loss deltas means the balanced model gains more from
calibration; a positive raw weighting contrast means balanced raw is worse.
Do not pool repeated predictions as if
they were independent songs, perform a t-test on the 20 splits, or label their
percentiles confidence intervals. The empirical spread is descriptive sensitivity
to splitting/refitting in this already observed dataset.

## Optional prespecified retrospective stress check

Also apply each saved model/calibrator to the previously scored 266 tracks / 200
artist IDs. These are disjoint from the development artists but their MAEST outcomes
were inspected before this extension. Results are therefore explicitly retrospective,
not a fresh test, not a second dataset, and never used for fitting or method choice.
MERT feature extraction is label-free with a separate pre-extraction provenance
receipt and a historical control. Freeze its completed feature hashes before stress
scoring. Main refits need not wait for this extraction. Report these scores separately.

## Boundaries

Same Jamendo source, four broad noisy observed tags, first-30-second audio, historical
selection and incomplete labels limit generalization. Artist aliases, encoder
pretraining overlap and project-external prior exposure remain unverified. Never
present this as cross-dataset validation. Earlier manifests, caches, reports and
application releases remain historical artifacts. Do not tune on the 266 and call
them unseen again. A future confirmatory study needs new data and its own protocol.

Relevant background: [sklearn class weights](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
[calibration survey](https://arxiv.org/abs/2112.10327),
[audio calibration](https://arxiv.org/abs/2206.13071), and
[imbalance correction study](https://arxiv.org/abs/2202.09101).
The existence of these studies rules out claiming that calibration or weighting
distortion is itself a new idea; this extension examines them in this music setup.
