# Post-hoc diagnosis of the unfavorable calibration split

## Status and question

Split 20260927 is selected because the previous studies already found that full-budget
sigmoid calibration worsened its macro Brier. It is therefore an outcome-selected,
post-hoc case study. This document fixes a bounded diagnostic workflow; it is not a
prospective preregistration. All five original splits are shown for context.

Question: where does the unfavorable change occur, and what observable differences
between calibration and evaluation are consistent with it? Do not claim to establish
its cause, label shift assumptions, distributional equality, or a superior new method.

## Fixed predictions and boundaries

Use only the saved classifiers' logits and fitted calibrator parameters from
outputs/calibration/20260926_v2. Reconstruct calibration probabilities using the same
maps, and verify evaluation probabilities against saved arrays. No model or calibrator
is fitted, no threshold is selected, and no songs or artists are removed from the main
results. All source, configuration and input hashes are recorded before this runner.
Alignment is verified against original IDs, broad targets and artist partitions.

Calibration-set losses are fitting-set diagnostics, with expected fitting optimism.
They must not be compared with evaluation losses as two independent validation estimates.
The diagnostics below use evaluation labels after results were known; they are not
tools for choosing a deployable calibration rule on the same evaluation set.

## Fixed descriptive summaries

For all five splits, both calibrators, and each label:

1. Report fit/calibration/evaluation track and artist counts and positive fractions.
   For calibration/evaluation report positive-artist counts, raw/calibrated mean
   probability, mean signed probability-minus-target error, Brier and binary log loss.
   Include saved slope/intercept and optimization status.
2. For raw scores in each subset and each observed outcome (0 and 1), report track and
   artist counts and the 10th, 50th and 90th quantiles. These are finite-sample descriptive
   values, not a test of equality of conditional score distributions.
3. Use five fixed equal-width RAW-score bins in both subsets. Each table includes all
   bins, including empty bins, track count, artist count, label frequency, raw mean
   score and recalibrated mean score. Bin membership is unchanged across methods.
   This permits a within-coarse-score-range comparison without hiding sample counts.
4. Report each label's evaluation Brier change and contribution to macro change (divide
   by four). Decompose the change exactly with d=q-p:
   mean[(q-y)^2-(p-y)^2] = mean[d^2] + 2*mean[d*(p-y)].
   Also split loss-change sums between positive and negative target rows, using the
   full evaluation size as denominator so that contributions add to the label total.
   This is an algebraic description, not a causal decomposition.

## Artist concentration and sensitivity

For every evaluation artist and method, save per-label and macro mean loss changes,
track counts and contribution to the full track-weighted macro change (sum of the
artist's per-track macro changes divided by all evaluation tracks). Contributions
must sum to the original result.

For each artist, calculate the remaining-row mean change after excluding that artist,
without refitting anything. Save ALL leave-one-artist-out values, their range and the
number below zero. This is a sensitivity diagnostic, not a new result obtained by
selectively removing an inconvenient artist. A robust sign does not establish statistical
significance. Artist rows can contain genuine variation or imperfect labels; this
analysis cannot identify label errors or blame artists.

Report the share of summed POSITIVE macro contributions from the top five positive
contributors. The denominator is positive contributions only, never the signed net
change; cancellation can otherwise create misleading percentages. Keep the full artist
table. Report artist-equal mean loss change as a separate descriptive estimand, since
it differs from the study's track-weighted target.

## Outputs and inference

Generate full tables, a focused comparison of calibration/evaluation mean scores and
observed fractions, label contributions across all five splits, and artist sensitivity.
No p-values, new confidence intervals or new method selection. Report patterns as
compatible explanations and distinguish them from identified mechanisms. Preserving a
bad split is part of the research outcome. A method informed by these observations
requires its own design and validation; reusing evaluation labels cannot make it independent.
