# Report figure provenance

These PNGs make the reports readable without the ignored `outputs/` directory.
The four historical figures were rendered from saved experiment results; their
inclusion does not add training runs, predictions, or new evaluation evidence.

| Asset | Report and scope | Generator | Saved inputs under `outputs/` |
| --- | --- | --- | --- |
| [baseline_f1.png](baseline_f1.png) | [Original baseline](../../RESULTS.md), 90 historical test tracks, exact tags | `report_results.py` | `baseline_metrics.json` |
| [feature_comparison.png](feature_comparison.png) | [Feature comparison](../../EXPERIMENTS.md), 90 validation tracks, exact tags | `report_comparison.py` | `feature_comparison/metrics.json` and both validation-prediction CSVs |
| [mfcc_diagnosis.png](mfcc_diagnosis.png) | [MFCC diagnostic](../../IMPROVEMENT.md), grouped training learning curve and 90-track validation | `report_improvement.py` | `improvement/mfcc_metrics.json` and `baseline_metrics.json` |
| [mert_comparison.png](mert_comparison.png) | [MFCC vs MERT](../../MERT_RESULTS.md), 90 validation tracks, exact tags | `report_mert.py` | `improvement/mfcc_metrics.json` and `improvement/mert_metrics.json` |
| [final_holdout_results.png](final_holdout_results.png) | [Final MAEST evaluation](../../FINAL_RESULTS.md), 239 holdout tracks, broad labels | `make_portfolio_figure.py` | See the generator and `data/final_metrics_summary.json` |

Run a report's generator from the repository root in its recorded environment
after preparing its required saved inputs. The four historical generators keep
their original local image outputs and copy the same bytes into this directory.
They render stored results without fitting classifiers or producing new model
predictions. Some generators also validate the original data and frozen hashes;
their other required inputs remain described in the respective reports.

Raw metrics, feature caches, and prediction files remain local. Only the report
images are included here; this does not change audio or model distribution.
