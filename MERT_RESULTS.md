# Frozen music representations: MFCC vs MERT

Development run: 2026-08-28.

## Question and scope

Does a frozen pretrained music representation improve this small four-genre tagging task over MFCC statistics?
The encoder, pooling, data, classifier grid and selection rule were fixed before viewing MERT outcomes.
See [protocol](experiments/IMPROVEMENT_PROTOCOL.md), [execution settings](experiments/mert_execution_plan.json) and [source review](experiments/MERT_REVIEW.md).

Both representations use the same 300 training tracks, the same three artist-grouped folds, six logistic-regression settings, and the same 90 validation tracks.
Choose the head by mean fold Macro AP; select thresholds on its training out-of-fold scores; refit on 300 tracks before validation prediction.
No new test predictions are made. The 90 validation tracks were previously used in development; these are NOT independent final test results.
MERT pretraining used Music4All and part of FMA; overlap with Jamendo has not been audited. Artist separation here applies to the project's downstream splits, not to the encoder's pretraining corpus.

## Representations and runtime

MFCC: 13 temporal means plus 13 standard deviations. MERT: resample the prepared first-30-second audio to 16 kHz, split into six 5-second chunks, mean-pool time and 12 transformer layers (excluding layer 0), then average the chunks: 768 features.
The encoder weights are frozen; only the scaler and four linear classifiers are fitted to this project's labels. No layer/crop/pooling search or encoder fine-tuning was performed.
This experimental MERT extractor currently requires at least 30 seconds. The original MFCC CLI still supports its existing shorter-input policy.

CPU pilot: loading 2.02s; two 30-second extractions 0.94s and 0.83s; maximum repeated-feature difference 0.
Full development extraction: 390 tracks; 318.6s in the recorded run (0 cached at start).
Platform: Python 3.13.15, Apple Silicon macOS 15.6.1, CPU with four PyTorch threads. GPU and other platforms were not benchmarked.
Runtime dependencies: torch 2.8.0, transformers 4.38.2; exact lock in requirements-mert-lock.txt. The original .venv was not changed.
All downloaded files were verified against the official pinned version. The local checkpoint was read with weights_only=True and strict state-dictionary matching; reviewed custom code runs locally without audio uploads.

## Same-protocol comparison

F1 policy: per-label thresholds selected for F1 on training OOF predictions.

| Representation | Selected C / weight | Mean CV Macro AP | Validation Macro AP | Micro precision | Micro recall | Micro-F1 | Macro-F1 | Track output coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MFCC | 0.1 / none | 0.5158 | 0.4242 | 0.410 | 0.701 | 0.517 | 0.442 | 1.000 |
| MERT | 0.01 / none | 0.6062 | 0.5331 | 0.517 | 0.619 | 0.563 | 0.522 | 0.911 |

Validation changes: Macro AP +0.1089, Micro-F1 +0.0463, Macro-F1 +0.0791.
AP measures ranking, not thresholded precision or classification accuracy. CV selection estimates are optimistic after choosing among six settings.
The MFCC row here is the newly selected MFCC head with OOF thresholds, not the original default demo with validation-tuned thresholds.

| Tag | Validation positives | MFCC precision | MERT precision | MFCC recall | MERT recall | MFCC AP | MERT AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| electronic | 46 | 0.511 | 0.673 | 1.000 | 0.717 | 0.737 | 0.768 |
| pop | 16 | 0.171 | 0.250 | 0.375 | 0.375 | 0.184 | 0.267 |
| ambient | 18 | 0.321 | 0.414 | 0.500 | 0.667 | 0.315 | 0.496 |
| rock | 17 | 0.538 | 0.643 | 0.412 | 0.529 | 0.461 | 0.602 |

!Development precision and recall (`outputs/improvement/mert_comparison.png`, generated locally)

## Precision-oriented operating point

Thresholds were chosen only from training OOF scores: >=80% precision, >=30% recall and >=10 emitted predictions per tag, then maximize recall among feasible choices.
Unsupported tags are suppressed under this policy. Precision is N/A when there are no predictions; stored zeros follow the metric function's zero-division convention.

| Tag | Feasible on training OOF? | Threshold | Validation TP | FP | FN | Predictions | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| electronic | False | disabled | 0 | 0 | 46 | 0 | N/A | 0.000 |
| pop | False | disabled | 0 | 0 | 16 | 0 | N/A | 0.000 |
| ambient | False | disabled | 0 | 0 | 18 | 0 | N/A | 0.000 |
| rock | True | 0.650 | 6 | 2 | 11 | 8 | 0.750 | 0.353 |

Micro precision: 0.750; micro recall: 0.062; track output coverage: 8/90 = 0.089.
The provisional development target additionally requires every tag to have >=80% precision, >=30% recall, >=5 validation predictions, plus >=50% track output coverage. It is not a guarantee of usefulness or a confidence bound.
**Provisional development target met: False.** Do not substitute overall accuracy, AP, or one successful tag for this statement.

## MERT's six classifier trials

| C | Class weight | Mean CV Macro AP | Fold scores |
| ---: | --- | ---: | --- |
| 0.01 | none | 0.6062 | 0.6295, 0.5941, 0.5951 |
| 0.01 | balanced | 0.6014 | 0.6297, 0.5838, 0.5908 |
| 0.1 | none | 0.5690 | 0.5693, 0.5587, 0.5789 |
| 0.1 | balanced | 0.5677 | 0.5674, 0.5572, 0.5786 |
| 1.0 | none | 0.5529 | 0.5407, 0.5444, 0.5736 |
| 1.0 | balanced | 0.5523 | 0.5400, 0.5446, 0.5725 |

## Limits and next decision

This finishes the predeclared candidate comparison. Do not continue tuning on these validation outcomes in this round.
The original predict.py and baseline bundle remain unchanged. Any eventual promotion must explicitly identify its operating policy and limitations.
A final reliability claim needs newly preselected independent tracks/artists, fixed model/thresholds, an overlap audit, and uncertainty estimates. The 90 old test tracks cannot be relabeled as unseen.
The small coverage-enriched sample, incomplete uploader genre labels, only four target genres, and first-30-second cropping limit generalization to arbitrary uploads.
Weight license: CC BY-NC 4.0 as identified by the model card. Distribution terms for project code, pretrained weights, classifiers, and example audio are handled separately.

## Reproduce locally

```bash
python3 -m venv .venv-improve
.venv-improve/bin/python -m pip install -r requirements-mert-lock.txt
.venv-improve/bin/python prepare_mert.py
.venv-improve/bin/python extract_mert.py pilot --device cpu
.venv-improve/bin/python extract_mert.py extract --device cpu
.venv-improve/bin/python improve_model.py mert --cache outputs/improvement/mert_features.npz
.venv-improve/bin/python report_mert.py
```

Requires the prepared 480-track project data and original baseline artifacts. No corpus audio is downloaded by the encoder setup; only public model files are retrieved.
If necessary, the downloader accepts --proxy with your existing proxy URL; it does not change system network settings. Do not disable TLS verification.
The upstream optional nnAudio warning is expected: CQT is disabled in this pinned checkpoint, so no nnAudio install is needed.

Sources: [pinned MERT model card](https://huggingface.co/m-a-p/MERT-v0-public/tree/e8413e398b3180ca488534aea68fd829402044a4), [PyTorch restricted checkpoint loading](https://docs.pytorch.org/docs/2.8/generated/torch.load.html).
