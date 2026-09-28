# Classifier / decision-rule ablation and training-amount diagnosis

This is a NEW exploratory experiment, fixed locally before its fitted models or
results. The previous automatic experiment failed its gate and remains unchanged.
The user authorized this next investigation. No human labels are required or
inferred, and no source tags are rewritten. A development improvement cannot be
called new-song confirmation or an application release.

## Pool and explicit role retirement

Combine verified MAEST caches in this order: original development 1206/469
tracks/artists; historical holdout 239/89; earlier validation 266/200; recent
application validation 204/150. Total: 1915 tracks / 908 artist IDs, pairwise
disjoint cohorts, float32 source vectors of width 2304. Training uses float64.
Verify IDs, cache hashes, source/extractor/encoder provenance and the same frozen
four-label ontology. Preserve the original tags and their mapped targets.

Before fitting, save a new role ledger naming all these rows development-only
for subsequent work. Preserve every original role and historical result. These
previous tests cannot later become untouched validation through a new split.
The future-new-song exclusion remains the complete old candidate-frame union
(2195 track IDs / 1008 artist IDs), including unsuccessful/unselected candidates.
No new audio acquisition, metadata-frame expansion or human answers occur here.

## One primary comparison, bounded hypotheses

The one primary comparison is `new_tuned` versus `baseline_original` at full
expanded development training budget. The new recipe was motivated by known
historical outcomes, not discovered on a new independent dataset.

Baseline heads: C=.001 logistic regression, ambient balanced class weights,
others unweighted; scaler fitted on each actual training subset. Ambient display
probability uses the analytical log(n_positive/n_negative) offset calculated
ONLY from that fit's labels. Original decisions remain raw probability >= exact
historical f1 threshold. The original representation/C/thresholds have themselves
been chosen previously using original development data, so this is not a wholly
unbiased first-use cross-validation estimate.

New heads: replace ONLY pop and ambient by fixed unweighted C=.0003 heads. No C
search, head screening or head fallback. Electronic/rock reuse the same fitted
baseline heads in all cells at a given training budget.

Full-budget factorial cells:

| Cell | Head recipe | Decision rule |
| --- | --- | --- |
| baseline_original | baseline | original |
| new_original | new | original, mapped to common probability scale |
| baseline_tuned | baseline | inner-selected rule |
| new_tuned | new | inner-selected rule |

For a baseline trained on some subset, its original raw cutoff t maps to
sigmoid(logit(t)+fit_only_offset). `new_original` uses that same corrected-
probability cutoff. This preserves a specified comparable rule across head
types, not a claim that either head is perfectly calibrated. Baseline originals
retain their exact raw comparison to avoid floating-point changes at boundaries.
Do not compare .375 raw directly with a new unweighted probability cutoff.

Within each outer training pool use four cohort/artist-grouped inner folds.
Every inner model gets its own scaler, class counts, offset and original cutoff.
Collect out-of-fold probability and original-decision arrays for each head.
For each focus label/head, candidate rules are its own original rule (which may
have a different mapped cutoff across inner fits) plus fixed constant corrected-
probability cutoffs .025 through .975 in .025 increments. Both heads use the SAME
baseline-original inner decisions as reference. Require FP<=reference FP,
recall>=reference recall-.03 and F1>=reference F1. Minimize FP, then maximize TP,
then prefer original over constant, then the higher constant cutoff. Use 1e-12
metric tolerance. If no rule is feasible, retain THAT SAME head's original rule
and record infeasibility; never silently substitute the baseline head.
Electronic/rock are not threshold-tuned. Probabilities for original/tuned cells
of the same head are exactly identical; threshold changes cannot change their
Brier, log loss or average precision.

## Data separation and learning curves

Use one five-fold outer assignment based on cohort and whole artist IDs, balanced
without reading targets. Each song appears in exactly one outer evaluation fold.
The same outer evaluation rows are used for ALL training budgets and all cells.
Train/inner-selection/evaluation artists must be disjoint at their boundaries.
No imputation, resampling, tuning or standardization sees outer evaluation labels.

The main learning curves use nested 40%, 70%, 100% artist subsets within each of
the four training cohorts. Within a cohort order artists by a fixed seeded hash,
take nested prefixes, and retain all songs of each selected artist.
Take ceil(fraction * cohort_artist_count) artists per cohort. Use all three
prespecified sampling seeds for 40/70%; 100% is identical and computed only once.
No label balancing or redraw is allowed. A single-class training subset stops
the run. The exact subsets and counts are saved before fitting.

Train the two FIXED head recipes at each partial budget; do not tune thresholds
or select heads on these curves. Primary curve measures are macro Brier, log loss
and average precision (AP); original-rule confusion counts are supplementary.
Full-budget models reuse the factorial original cells. Compare each curve on
the SAME outer-held songs, also reporting each source cohort separately.
These are related subsamples/folds, not independent experimental replications;
report all three prefix draws and their range, not an independent-replicate CI.

Also train both recipes on original-development-only artists remaining in each
outer training pool, once per outer fold, and score the same held songs. This
practical comparison changes BOTH training quantity and source composition. Do
not describe it as a pure causal effect of either quantity or diversity. It is
not a new test of the unchanged deployed application, which used all 1206 rows.

Factorial head contrasts, decision-rule contrasts and their interaction are
descriptive. Since each head separately selects its rule, there is no unique
additive percentage of the total change attributable to each component. Do not
use the best cell, budget, cohort or sampling seed to replace the primary method.

## Gate and stopping rules

Pool full-budget outer predictions once. `new_tuned` passes only if all hold
against `baseline_original`: pop+ambient FP falls at least 10%; each focus FP
does not increase; micro recall loses at most .03 and each label at most .05;
micro F1 does not fall; macro Brier and binary log loss do not increase; exact
electronic/rock controls and every group/row boundary pass. These are application
tradeoffs, not a statistical significance test. Report failed conditions.
Both primary arms are refitted on expanded training subsets; this gate compares
recipes, not the actual installed v1.1 artifact. Future confirmation must compare
the final frozen candidate against the real installed baseline and explicitly
specify whether electronic/rock remain copied original heads or are retrained.

If a DIFFERENT diagnostic cell looks better, it remains a hypothesis for another
separately frozen workflow; do not promote it after seeing these outer answers.
This stage does not fit a final deployment artifact or fetch new songs regardless
of the development gate. A promising result supports freezing a final procedure
and a separate fresh-song confirmation later; a negative result is retained.
No original test, application artifact or prior source file is overwritten.

## Reporting and verification

Keep pooled and fold-level TP/FP/FN/precision/recall/F1, probability scores and AP,
source-cohort metrics, training row/artist counts, nested subset IDs, choices,
models and prediction hashes. Undefined AP with no source-positive examples is
null, never a reason to drop rows. Confusion counts are label cases, not unique
songs. Save all failures and verify stored predictions/model identities without
refitting. Do not claim that extra data guarantees improvement or that proxies
establish a song's true style. The fixed 30-second feature representation,
possible label omissions, source sampling bias, artist aliases and encoder
pretraining exposure remain limitations.
