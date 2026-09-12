# Evaluation review supplement

Added 2026-09-12. This is a post-hoc analysis of saved predictions.
The original 239-track results remain the primary historical record. No model was fitted
or tuned for this analysis, and the reduced subsets are not new independent tests.

Recompute with `python evaluate_release.py --output outputs/review_audit.json`.
All original metrics for both policies, their track-bootstrap intervals, and the
overall/per-label listening counts were checked against the published archived results.

## Historical exposure

The 239 holdout tracks cover 89 artists excluded from final fitting. Of these,
40 artists also occurred in the previously observed historical test, accounting for
150 holdout tracks. `track_0543400` was a learning/decoding sample before final evaluation.
The original 480-track subset excluded all three learning tracks; expansion reintroduced
this one. The original dataset document describes that earlier subset only.

## Artist-cluster uncertainty

Sample 89 artists uniformly with replacement; retain all their tracks each time. Metrics pool tracks (not equal artist weighting); sample track count can vary. No-positive AP/recall and no-prediction precision are zero, matching archived conventions. All replicates retained; counts reported. Fixed model only, no training uncertainty.

Each policy uses 2,000 replicates and percentile 95% intervals; seeds are 20260830
(primary) and 20260831 (secondary). Artist IDs are sorted before seeded sampling.
This supplements the final protocol's track bootstrap and documents a change from
the earlier expansion protocol's promised artist-grouped uncertainty. It does not
address source-label errors, pretraining overlap, or selection of this convenience subset.

| Policy / metric | Estimate | Original track interval | Artist-cluster interval |
| --- | ---: | --- | --- |
| f1 / micro_precision | 0.498 | 0.444–0.553 | 0.392–0.587 |
| f1 / micro_recall | 0.755 | 0.695–0.818 | 0.663–0.828 |
| f1 / micro_f1 | 0.600 | 0.549–0.651 | 0.502–0.679 |
| f1 / macro_ap | 0.594 | 0.538–0.671 | 0.516–0.686 |
| f1 / track_output_coverage | 0.862 | 0.820–0.904 | 0.792–0.917 |
| precision_target / micro_precision | 0.678 | 0.577–0.772 | 0.565–0.805 |
| precision_target / micro_recall | 0.289 | 0.232–0.350 | 0.220–0.389 |
| precision_target / micro_f1 | 0.405 | 0.336–0.471 | 0.323–0.515 |
| precision_target / macro_ap | 0.594 | 0.540–0.671 | 0.512–0.688 |
| precision_target / track_output_coverage | 0.364 | 0.305–0.427 | 0.277–0.450 |

## Previously handled sample: sensitivity

Both exclusions were chosen after the historical evaluation was available. All four
labels, thresholds, model scores and heads stay fixed. Results below use the primary policy;
the machine-readable supplement also includes the secondary policy.

| Subset | Tracks | Artists | Micro precision | Micro recall | Micro F1 | Macro AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 239 | 89 | 0.498 | 0.755 | 0.600 | 0.594 |
| exclude_learning_tracks | 238 | 89 | 0.498 | 0.754 | 0.600 | 0.593 |
| exclude_learning_artists | 237 | 88 | 0.498 | 0.752 | 0.600 | 0.593 |

These exclusions cannot remove the broader historical artist exposure or establish
that earlier handling had no influence on project decisions.

## Listening order

The original interface hid source targets and model results, but each label's five
source-positive questions preceded its five source-negative questions. Someone familiar
with the protocol could infer the source target from order. This is an order-structured
single-reviewer audit, not a fully concealed blind test. The recorded 29/36 agreement
remains descriptive; actual influence from order cannot be established from these records.
Future audits should shuffle new questions globally with a recorded seed before listening.
The historical query order and human answers have not been rewritten.
