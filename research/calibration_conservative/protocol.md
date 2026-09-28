# Conservative sigmoid calibration: frozen exploratory protocol

## Question and provenance

Can calibration-only grouped validation reduce harmful probability adjustments while
retaining useful ones? Earlier evaluation results, including the unfavorable split
20260927, motivated this proposal. This is an exploratory method-development study,
not a new independent confirmation or an external preregistration. Rules below are
fixed before computing the new policy's evaluation outcomes. All five original outer
splits are reported and 20260926 remains primary. The known bad split is not promoted
to primary and does not receive a special selection rule.

## Fixed foundation

Use each split's original frozen classifier/scaler, calibration logits and broad labels,
evaluation logits, and original identity/sigmoid/temperature predictions. No encoder or
classifier retraining, feature selection, threshold selection, or new audio. Restore
the original sigmoid parameterization and optimizer settings unchanged. All input
IDs/labels/artists and source hashes must match. The old 239/90-track datasets remain
outside this experiment.

## Inner grouped validation, entirely within calibration

For each outer split, sort its calibration artist IDs, permute them with numpy PCG64
using SeedSequence([2026092603, outer_seed]), and split into five contiguous chunks by
np.array_split. Each chunk contains whole artists; all of their songs form one inner
validation fold. All other calibration artists form that fold's fitting subset.
There is no label balancing or search over fold seeds. The same folds apply to all labels.
Log each fold's training/validation IDs, artists and positive/negative counts.

For each label and inner fold, fit sigmoid on that fold's fitting logits and labels,
then predict the held-out calibration rows. Every calibration song receives exactly
one out-of-fold sigmoid probability. The base classifier never used calibration rows
for its training. Inner validation labels enter only the selection score, not the
sigmoid fit producing their probabilities. Held-out folds need not contain both classes;
the fitting subset must. Complete all five inner fits and log every status.

## Fixed selection heuristic

For each label, let p be raw calibration probabilities and q be out-of-fold sigmoid
probabilities. Candidate r(lambda)=(1-lambda)*p+lambda*q, with lambda in
{0,0.25,0.5,0.75,1}. Lambda zero is identity and one is full adjustment. Compute
track-weighted OOF Brier for all candidates. Choose the smallest lambda within an
absolute 1e-12 tolerance of the minimum as the best candidate.

Let d_i be that best candidate's squared loss minus identity's squared loss on row i,
N the number of calibration songs and G the number of calibration artists. Define
u_g=sum_{i in g}(d_i-mean(d)). The paired artist-cluster SE proxy is
sqrt(G/(G-1) * sum_g(u_g^2) / N^2).
Set threshold = Brier(best) + SE_proxy. Among candidates with Brier no greater than
threshold plus 1e-12, choose the smallest lambda. No SE multiplier tuning. When identity
is already best, d is zero and identity is retained. The heuristic measures paired loss
variation and favors smaller adjustments when the apparent improvement is modest.

This is a **paired-SE shrinkage heuristic inspired by the one-standard-error principle**,
not the textbook foldwise rule, a significance test, a confidence interval or a guarantee
of non-degradation. Inner fits share training observations and the best candidate is
data-selected. The SE proxy ignores those dependencies and uncertainty from refitting.
The selected OOF loss is a model-selection statistic, not an unbiased reported test score.

Refit sigmoid on the full calibration pool after selecting lambda, save its parameters,
and apply the selected blend to evaluation logits. Full-pool sigmoid probabilities
must match the original full sigmoid predictions within 1e-12. OOF selection evaluates
sigmoids trained on about 80% of calibration artists, so transfer to the full-pool map
is itself an approximation. Do not average evaluation predictions from the inner fits.
The four binary probabilities are never normalized to sum to one.

If any inner training subset lacks two classes, an optimizer fails, or generated
probabilities are nonfinite, retain identity for that label and record the reason;
never redraw folds. If the full-pool fit fails, also retain identity for that label.
Bound hits alone are recorded but retained. Identity fallback is an explicit part of
this policy, not a silent replacement of a failed method. Programming/alignment errors
abort. No evaluation labels, probabilities or metrics enter policy selection.

## Execution boundary and outcomes

Separate freeze, select and evaluate commands. Freeze all folds and source/input hashes;
then generate and save all OOF arrays, candidate scores, parameters and chosen strengths
for all five outer splits using only calibration data. Only after every selection is
complete generate new evaluation probabilities and scores. The selection function has
no evaluation-data argument.

Report identity, old full sigmoid, old temperature and the new conservative sigmoid on
identical rows. Primary metric: equally label-weighted mean binary Brier. Auxiliary:
per-label Brier, binary log loss, 5/10 equal-width reliability bins, AP and optimizer /
fallback / bound-hit counts. Keep unfavorable outcomes and all selected strengths.
No evaluation-driven change of lambda grid, inner seed, selection rule or label policy.

Primary-split uncertainty: 2000 paired evaluation artist-cluster bootstrap draws with
seed 2026092604. Report conservative-minus-identity and conservative-minus-full-sigmoid
Brier/log-loss differences, macro and by label. Reuse the SAME resampling indices for
both comparisons. Intervals are pointwise percentile intervals conditional on the base
classifier, calibration data and selected policy. They do not include selection/refitting
uncertainty, historical method development or multiple-comparison correction. Other
outer splits are descriptive robustness checks, not independent replicates for a t-test.

## Sources and limits

The one-standard-error simplicity principle is described in the Stanford STATS 202 notes:
https://web.stanford.edu/class/stats202/notes/Resampling/Kfold-CV.html
The paired clustered tolerance above is a specified adaptation, not a claim that those
notes validate this procedure. Group separation follows scikit-learn's grouped-CV guidance:
https://scikit-learn.org/stable/modules/cross_validation.html
Dependence in CV uncertainty estimates is discussed by Bates, Hastie and Tibshirani:
https://arxiv.org/abs/2104.00673

All earlier limitations remain: history-informed method design, enriched samples,
potentially incomplete labels, first-30-second/whole-track mismatch, unaudited pretraining
overlap and limited artists. Identity can be chosen too often and useful improvements
lost; retained calibration can still harm a different sample. This candidate is not
automatically a deployment change or evidence of novel methodology. New independent
data and a separately fixed validation plan would be needed for stronger claims.
