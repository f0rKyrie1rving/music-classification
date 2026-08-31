# Performance improvement: development checkpoint

Run date: 2026-08-28.

## What ran

Three artist-grouped folds on the original 300 training tracks; six predeclared logistic-regression settings.
The scaler is fitted separately inside each fold. Model selection uses mean fold Macro AP; thresholds use out-of-fold training scores.
The existing 90 validation tracks are evaluated without retuning their thresholds in this round. They were used in earlier development, so this is NOT a fresh blind test.
No new test predictions were made. The original baseline model, feature code, manifest and results remain unchanged.
Protocol: [IMPROVEMENT_PROTOCOL.md](experiments/IMPROVEMENT_PROTOCOL.md); exact splits: [improvement_plan.json](experiments/improvement_plan.json).

## Bounded parameter comparison

C controls regularization strength: a smaller C penalizes large coefficients more strongly. Balanced weighting changes how errors on rare labels affect training.

| C | Class weighting | Mean fold Macro AP | Fold 1 | Fold 2 | Fold 3 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.01 | none | 0.5057 | 0.4876 | 0.5478 | 0.4817 |
| 0.01 | balanced | 0.5067 | 0.4900 | 0.5457 | 0.4844 |
| 0.1 | none | 0.5158 | 0.5255 | 0.5154 | 0.5065 |
| 0.1 | balanced | 0.5126 | 0.5287 | 0.5116 | 0.4975 |
| 1.0 | none | 0.4891 | 0.4970 | 0.4820 | 0.4884 |
| 1.0 | balanced | 0.4771 | 0.5021 | 0.4669 | 0.4624 |

Selected C=0.1, class_weight=None. This is a bounded development choice, not a statistical superiority claim.
The winning CV score is optimistic after selecting among six settings. Fold-to-fold variation is visible; do not interpret it as a confidence interval.

## Learning curve for the original fixed C=1 / balanced setting

Nested subsets of training ARTISTS are used inside each fold. The held-out artists stay fixed within that fold.
Macro AP measures ranking, not precision after applying a threshold and not classification accuracy.

| Mean training tracks (range) | Training Macro AP | Held-out Macro AP | Held-out fold SD |
| --- | ---: | ---: | ---: |
| 48 (46–50) | 0.9505 | 0.3984 | 0.0630 |
| 100 (95–107) | 0.7554 | 0.4208 | 0.0565 |
| 200 (194–209) | 0.6458 | 0.4771 | 0.0177 |

Held-out mean AP improves with more training artists in this small experiment, but one fold is non-monotonic.
This supports investigating data quantity/diversity; it does not establish a sample count that would yield 80% precision.
High training scores on tiny subsets with much lower held-out scores show sensitivity to limited data. The curve does not isolate feature quality from label noise or sampling bias.

!Grouped learning curve and validation ranking (`outputs/improvement/diagnosis.png`, generated locally)

## Existing validation set (90 tracks)

| Model / threshold source | Macro AP | Micro-F1 | Macro-F1 |
| --- | ---: | ---: | ---: |
| Original baseline / same validation set | 0.4085 | 0.5260 | 0.4894 |
| Selected MFCC head / training OOF | 0.4242 | 0.5171 | 0.4424 |

AP comparisons use the same validation tracks and do not depend on thresholds. F1 also reflects the different threshold-selection data, so this is NOT a pure classifier-only F1 comparison.

Selected head with training-OOF F1 thresholds:

| Tag | Threshold | TP | FP | FN | Precision | Recall | Predictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| electronic | 0.150 | 46 | 44 | 0 | 0.511 | 1.000 | 90 |
| pop | 0.250 | 6 | 29 | 10 | 0.171 | 0.375 | 35 |
| ambient | 0.250 | 9 | 19 | 9 | 0.321 | 0.500 | 28 |
| rock | 0.400 | 7 | 6 | 10 | 0.538 | 0.412 | 13 |

Micro precision: 0.410; micro recall: 0.701; track output coverage: 1.000.
Coverage means at least one of the four tags is emitted; it does not mean the emitted tags are correct.

## 80% precision target: NOT met

For the selected MFCC head, no tag has a threshold in the predeclared grid that simultaneously meets 80% precision, 30% recall and 10 emitted predictions on the 300 training-OOF tracks.
The strict policy therefore marks all four tags unsupported and emits none. Its precision is undefined (N/A), recall and coverage are zero; stored numeric precision zeros follow the metric library's zero-division convention.
This abstention is a failed operating point, not a successful 80% model and not evidence that all music has no genre.
The result applies to this selected representation/head, this finite threshold grid and this data subset. It does not prove that 80% is impossible with other data or features.

## Decision at the MFCC checkpoint

Do not promote the selected MFCC head. The ranking improvement on validation is small and the operating trade-off is not adequate for the stated target.
MERT-v0-public is the one planned frozen encoder; its model card lists Music4All and part of FMA for pretraining. Training-corpus overlap with Jamendo has not been audited.
At the initial MFCC checkpoint, model downloads failed and no MERT results were available. This is a historical description of that checkpoint.
Connectivity has since been resolved using the existing system proxy; the separately completed [MERT comparison](MERT_RESULTS.md) contains its measured results.
The original test set remains historically observed. Any final performance claim still needs newly preselected independent tracks/artists, a fixed model and thresholds, and uncertainty reporting.

## Reproduce the completed diagnostic

```bash
.venv/bin/python improve_model.py mfcc
.venv/bin/python report_improvement.py
.venv/bin/python -m unittest discover -s tests -v
```

Requires existing prepared data and baseline artifacts. The split/parameter plan is already frozen; do not rerun the `freeze` action.
Local raw results: metrics (`outputs/improvement/mfcc_metrics.json`, generated locally), scores (`outputs/improvement/mfcc_scores.npz`, generated locally).

Sources: [MERT model card](https://huggingface.co/m-a-p/MERT-v0-public), [MERT runtime warning](https://github.com/yizhilll/MERT), [scikit-learn learning curves](https://scikit-learn.org/stable/modules/learning_curve.html).
