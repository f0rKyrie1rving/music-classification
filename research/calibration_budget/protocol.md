# Calibration-budget sensitivity protocol, version 1

## Question and status

How do the probability-quality changes and sensitivity to calibration sample selection
vary with the number of calibration artists, while classifier and evaluation rows stay
fixed? This protocol is written after the first study's five-split outcomes were known
and before computing new subset-calibration outcomes. It is an exploratory extension,
not a fresh confirmatory test or an externally registered protocol. It does not establish
a universal minimum sample size, a deployment policy or the superiority of a new method.

## Inputs and isolation

Reuse all five frozen splits from outputs/calibration/20260926_v2. Keep 20260926 primary.
For each split reuse its saved full-precision calibration/evaluation logits and original
raw probabilities, with ID, label and artist alignment checked against the frozen split
and manifest. Each split's classifier and scaler remain fixed. Never retrain a classifier
on a calibration row. No encoder, label definition, weight, calibration constraint or
optimizer changes. The historical 239/90 sets remain outside this experiment.

The freeze records source and input hashes, dependency versions, sampling indices,
track IDs, artists, outcome counts, timestamps, base Git commit and working-tree status.
Separate CLI steps freeze, fit and score the entire study. The fitting function accepts
calibration labels and evaluation logits, but no evaluation labels. All fits and
predictions are saved before any new evaluation score is produced. Numerical assertions
may compare the full-budget probability vectors with the saved first study's vectors.

## Sampling rule, fixed without new outcomes

Each calibration pool contains 94 artists. Fractions 0.25, 0.50, 0.75, 1.00 correspond
to ceil(fraction * 94), hence 24, 47, 71, 94 artists. Retain every track of each artist.
Track counts vary, so the controlled budget is artists, not exactly a quarter of songs.

For each outer split, make 30 independent seeded permutations of sorted calibration
artist IDs. Use numpy SeedSequence([2026092601, outer_seed, repeat_index]) and PCG64
through default_rng. Prefixes of 24, 47 and 71 artists are nested within each repeat.
Keep selected rows in original calibration row order. The full pool is identical for
all repeats and is fitted once per outer split. Total: 5 * (30 * 3 + 1) = 455 subsets,
910 method/subset combinations and 3640 binary calibration fits if all labels qualify.

Do not balance, redraw, reject or select subsets using labels or scores. Record positives,
negatives, positive artists, selected artist order and IDs. If a label has only one
outcome, report that label's fit as unavailable; retain the subset and other labels.
If optimization fails, save the failure and continue the scheduled comparisons without
fallback. Bound hits remain results and are counted. Macro results are unavailable if
any of four labels failed. Programmer errors abort with a failure record rather than
being disguised as statistical failures.

## Methods and outcomes

Reuse the first study's identity baseline and the same monotone sigmoid and binary
temperature fits, optimizing unweighted log loss with its fixed bounds, tolerances and
initialization. Retain temperature despite its previous unfavorable outcome. Do not add
ambient-only calibration or choose the best method after scoring. Predictions are
independent binary probabilities, not softmax outputs.

Primary: calibrated-minus-identity macro Brier on exactly the same evaluation rows.
Secondary: macro log loss and label-level Brier/log loss. Save full evaluation probability
arrays, optimizer parameters and statuses for every subset. Reuse epsilon=1e-12.

For each split/method/budget, report completed/attempted fits, number with lower Brier,
median difference, 10th/90th percentiles, minimum/maximum, median/range of song counts,
and bound hits. Plot Brier differences against calibration artist count. The 10th/90th
bands describe random calibration-subset sensitivity conditional on this fixed pool,
classifier and evaluation set. They are NOT confidence intervals and do not include
evaluation resampling, classifier refitting or population shift. A full-budget point has
only one fit, not 30 independent copies and not evidence of zero population uncertainty.

Also compare 25% versus 100% and adjacent budgets using matched nested repeats. When
100% is involved, subtract each repeat's result from the one common full-budget result.
Report counts and medians descriptively; do not interpret repeated use of a shared
endpoint or evaluation tracks as independent data. Do not use t-tests or pool predictions
across overlapping outer splits. Show all outer splits separately; descriptive counts
across the five splits carry no population-frequency interpretation.

## Interpretation and next steps

Previously known outcomes motivated this budget question; evaluation labels have already
been inspected. Do not claim an unbiased chosen budget or method. Learning-curve trends
can be nonmonotone because small calibration samples may mismatch the evaluation sample.
More data can stabilize an unfavorable adjustment. An effect on ambient alone does not
justify replacing the all-label method after the fact. Establishing a practical budget
or changing the deployed model requires a separate decision criterion and new validation.

Original study inputs, source files and results are read-only. Create a new directory;
refuse overwrites. No new audio, model download or global environment changes are needed.
