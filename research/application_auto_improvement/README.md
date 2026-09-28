# Automatic development experiment

This workflow does not require listening labels. It tests a bounded change to
the pop/ambient logistic heads and decision thresholds using the original 1,206
development songs. Read [the fixed protocol](protocol.md) before interpreting
results. The existing 239-, 266- and 204-song evaluations are not used for tuning.

Freeze the source, input hashes and nested artist folds before fitting:

```sh
.venv-improve/bin/python -m research.application_auto_improvement.run freeze \
  --out outputs/application_auto_improvement/20260927_v1
.venv-improve/bin/python -m research.application_auto_improvement.run run \
  --out outputs/application_auto_improvement/20260927_v1
.venv-improve/bin/python -m research.application_auto_improvement.run verify \
  --out outputs/application_auto_improvement/20260927_v1
```

The core and pipeline keep fitting/selection data separate from outer-fold
evaluation. Artist-disjoint outer results estimate the development procedure;
they are not a new independent test of the final full-data classifier.

The pass/fail conditions are application tradeoffs chosen before fitting. A
failure is retained and ends this candidate workflow. A pass permits a final
development refit and a separately frozen fresh-song confirmation; it does not
replace the application automatically or establish statistical significance.

All new outputs live under a new directory. No old model, feature cache, source
tag, human-review response or evaluation file is overwritten. Source audio and
third-party encoder weights are not distributed by this workflow.

Methodological context: official scikit-learn documentation on
[nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
and [grouped data](https://sklearn.org/stable/modules/cross_validation.html).
