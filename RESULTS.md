# First MFCC baseline

Actual results on the custom MTG-Jamendo subset.

## Protocol

Train / validation / test: 300 / 90 / 90 tracks.
Artist IDs do not overlap across partitions. The model uses 13 MFCC means and 13 standard deviations.
Standardization is fitted on training data only. Each tag threshold is selected on validation data, then fixed for testing.
The comparison baseline predicts the training-set tag frequencies and uses the same validation threshold procedure.

## Test results

| Method | Micro-F1 | Macro-F1 | Macro AP |
| --- | ---: | ---: | ---: |
| Tag-frequency baseline | 0.425 | 0.418 | 0.269 |
| MFCC + logistic regression | 0.480 | 0.488 | 0.451 |

| Tag | Test positives | Precision | Recall | F1 | AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| electronic | 35 | 0.408 | 0.886 | 0.559 | 0.532 |
| pop | 25 | 0.288 | 0.600 | 0.390 | 0.312 |
| ambient | 15 | 0.226 | 0.800 | 0.353 | 0.374 |
| rock | 22 | 0.722 | 0.591 | 0.650 | 0.584 |

### Interpretation

The observed Macro-F1 difference is +0.069. Macro-F1 averages the four per-tag F1 scores; it is not classification accuracy.
F1 balances precision and recall; AP summarizes the ranking of positive examples across thresholds.
Per-tag results matter: an average improvement does not mean every tag improved. No statistical significance or population-wide gain is claimed.
The frequency baseline assigns the same score to every track for a given tag; it does not use audio.
After validation threshold selection, it always outputs: electronic, pop, ambient, rock.

| Tag | Validation-selected threshold |
| --- | ---: |
| electronic | 0.30 |
| pop | 0.40 |
| ambient | 0.20 |
| rock | 0.70 |

!Per-tag F1 (`outputs/baseline_f1.png`, generated locally)

## Two error examples

Selected by the number of mismatched tags (descending), then track ID; all predictions remain available in the CSV.

- [track_1060601](http://www.jamendo.com/track/1060601): dataset tags = pop, rock; predicted = electronic, ambient.
- [track_1118603](http://www.jamendo.com/track/1118603): dataset tags = electronic, ambient; predicted = pop, rock.

## Limitations

This is a small, coverage-enriched subset of four archive shards, not the official benchmark distribution.
Uploaders' tags may be incomplete; a missing tag is not proof that a musical quality is absent.
Only the first 30 seconds are analyzed. MFCC summary statistics discard time order and do not fully describe rhythm or harmony.
Scores are not calibrated confidence. Results do not establish performance on arbitrary commercial music.
No MFCC-count or classifier hyperparameter search has been performed. Test results were not used to tune this baseline.

See [dataset protocol and licenses](data/DATASET.md), raw metrics (`outputs/baseline_metrics.json`, generated locally), and all test predictions (`outputs/test_predictions.csv`, generated locally).
