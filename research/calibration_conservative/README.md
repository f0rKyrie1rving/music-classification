# Conservative calibration experiment

This experiment tests whether internal validation can choose a smaller probability
correction when the evidence for full sigmoid calibration is weak. It is exploratory:
the original evaluation results had already motivated the method. It does not change
the application's trained classifier or its feature cache.

Read [protocol.md](protocol.md) for the fixed method and limitations. Selection uses
only the original calibration pools, with whole artists assigned to five folds.
Each label selects a blend from 0%, 25%, 50%, 75%, and 100% correction. The paired
artist-cluster standard-error tolerance is a shrinkage heuristic, not a statistical
guarantee against worse predictions.

## Reproduce

Run from the repository root with the existing `.venv-improve` environment. The
original frozen experiment in `outputs/calibration/20260926_v2` is required. For a
rerun, substitute a **new, unused** output directory at every step; stages refuse to
overwrite existing artifacts.

```sh
.venv-improve/bin/python -m unittest research.calibration_conservative.test_conservative -v
.venv-improve/bin/python -m research.calibration_conservative.run freeze --out outputs/calibration_conservative/20260926_v1
.venv-improve/bin/python -m research.calibration_conservative.run select --out outputs/calibration_conservative/20260926_v1
.venv-improve/bin/python -m research.calibration_conservative.run evaluate --out outputs/calibration_conservative/20260926_v1
.venv-improve/bin/python -m research.calibration_conservative.run bootstrap --out outputs/calibration_conservative/20260926_v1
.venv-improve/bin/python -m research.calibration_conservative.verify_results --out outputs/calibration_conservative/20260926_v1
.venv-improve/bin/python -m research.calibration_conservative.report --out outputs/calibration_conservative/20260926_v1 --destination docs/calibration_conservative
```

All five selections must finish before the evaluation stage begins. Source and
input hashes are frozen, so changing the implementation requires a new experiment
version. The version name identifies the study started on September 26; timestamps
inside the artifacts record the actual execution times.

## Retained artifacts

- `folds.json`: complete artist-isolated inner-fold assignments and counts.
- `selection/`: out-of-fold probabilities, all fitted parameters, candidate losses,
  selected strengths, and explicit failure reasons. Failed inner predictions remain
  `NaN` and trigger raw-probability fallback for the affected label.
- `evaluation/`: all five original evaluation pools, all four methods, label metrics,
  macro metrics, and per-song probabilities.
- `bootstrap/`: 2,000 paired artist resamples for the primary split, comparing the
  conservative method with raw and full sigmoid probabilities using the same draws.
- `verification.json`: independent reconstruction of choices, predictions, sklearn
  Brier/log-loss checks, and bootstrap draws. Unit tests are run separately.
- `source/` and `freeze.json`: exact source snapshots and input hashes.

The nine unit tests cover grouped partitions, the standard-error calculation,
strength selection, single-class and optimizer failures, probability bounds,
independent labels, and exclusion of held-out answers from their own inner fits.
Bootstrap intervals are conditional on the fitted and selected models; they do not
account for method development, selection uncertainty, or overlapping outer splits.

See the generated [research report](../../docs/calibration_conservative/REPORT.md)
and [Chinese explanation](../../docs/calibration_conservative/SUMMARY_ZH.md).
