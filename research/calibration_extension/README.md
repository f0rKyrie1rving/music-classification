# Calibration extension

Three questions are evaluated together: the ambient class-weighting mechanism,
transfer across MAEST/MERT-v0/MERT-v1 representations, and variability when scalers,
classification heads and calibration are refitted on 20 paired artist splits.
The final reported run is `outputs/calibration_extension/20260927_v2`.

Read the [frozen protocol](protocol.md), [implementation history](EXECUTION_NOTES.md),
[Chinese explanation](../../docs/calibration_extension/SUMMARY_ZH.md), and
[quantitative report](../../docs/calibration_extension/REPORT.md).

## Inputs and scope

This extension runs within the original project; it is not yet a standalone package
that reproduces everything from a clean clone. It requires the three previously
verified 1,206-track feature caches, their provenance files, historical manifests,
and original extraction/model receipts. It checks IDs, hashes, shape and finite values
before fitting. No original training runner is imported or modified.

The optional-but-completed retrospective stage additionally uses the earlier 266
songs and local MERT model files/audio. It downloads nothing. `cache.py` verifies the
original receipts and audio hashes, freezes its extraction procedure, reconstructs
a historical control, then creates new MERT caches in a separate output directory.
Those songs are already observed project data; this is not cross-dataset validation
or a fresh holdout. The 239-song historical application holdout is not rescored.

## Run

Use the existing `.venv-improve` environment (numeric runtime versions are stored in
each freeze receipt). Run from the repository root. For a genuinely new execution,
choose a new output directory; never overwrite the reported run or frozen inputs.

```sh
.venv-improve/bin/python -m unittest research.calibration_extension.test_core research.calibration_extension.test_cache -v
.venv-improve/bin/python -m research.calibration_extension.run freeze --out outputs/calibration_extension/reproduction_01
.venv-improve/bin/python -m research.calibration_extension.run fit --out outputs/calibration_extension/reproduction_01
.venv-improve/bin/python research/calibration_extension/cache.py audit
.venv-improve/bin/python research/calibration_extension/cache.py extract
.venv-improve/bin/python -m research.calibration_extension.run stress --out outputs/calibration_extension/reproduction_01
.venv-improve/bin/python -m research.calibration_extension.run summarize --out outputs/calibration_extension/reproduction_01
.venv-improve/bin/python -m research.calibration_extension.verify_results --out outputs/calibration_extension/reproduction_01
.venv-improve/bin/python -m research.calibration_extension.report --out outputs/calibration_extension/reproduction_01 --destination docs/calibration_extension_reproduction
```

`cache.py` uses its fixed dedicated extraction directory and verifies/reuses completed
features. It does not replace existing caches. All head fits are made from scratch
in a new main run; resuming an existing main run accepts only complete hashed runs.
An incomplete/failed model run stops rather than substituting another random seed.

## Outputs and interpretation

- `runs/<seed>/<representation>/<weighting>/`: non-executable NumPy head arrays,
  calibration parameters, inner-fold OOF predictions, main and retrospective
  predictions, optimizer statuses, per-label metrics and file hashes.
- `analysis/`: all split-level scores, calibration parameters, ambient mechanism
  contrasts, no-SE ablation, descriptive aggregate distributions and win/tie/loss
  counts. No independent-replicate confidence interval is constructed.
- `verification.json`: independent reconstruction of saved predictions and all
  statistics, using separate formulae and sklearn metric checks. It also checks the
  exact unchanged-label negative controls and retrospective provenance chain.
- `docs/calibration_extension/`: readable reports, figures and compact numerical
  outputs. Audio, upstream model weights and virtual environments are not exported.

Here “retraining” means refitting the scaler, linear heads and calibrators; pretrained
audio encoders remain frozen. Twenty overlapping splits are not twenty independent
datasets, and a Brier decrease is not an accuracy increase. Fixed C and pooling
settings allow a controlled comparison but not a best-tuned encoder ranking.
No method is selected for application deployment from these retrospective results.
