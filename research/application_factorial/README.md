# Classifier, decision-rule and training-data diagnosis

This exploratory study reuses 1,915 already observed tracks from 908 artist IDs.
It separates fixed classifier changes from inner-selected decision rules, and
compares nested training budgets on identical outer-held tracks. It never changes
the installed application or acquires a new validation cohort.

The [protocol](protocol.md) and [configuration](config.json) are frozen before
fitting. Prior evaluated cohorts are explicitly retired to development-only use;
the full historical candidate-frame exclusion remains 2,195 tracks / 1,008 artists.

From the repository root, using the existing research environment:

```sh
.venv-improve/bin/python -m unittest discover -s tests -p 'test_application_factorial*.py' -v
.venv-improve/bin/python -m research.application_factorial.run freeze
.venv-improve/bin/python -m research.application_factorial.run run
.venv-improve/bin/python -m research.application_factorial.run verify
.venv-improve/bin/python docs/application_factorial/independent_verify.py
```

Freeze and run are one-time operations. Completed outputs can be replayed without
refitting. Failed or interrupted fits remain recorded and cannot be silently
retried. Source snapshots, data-role and exclusion ledgers, every training subset,
model arrays, inner choices and outer predictions are retained in
`outputs/application_factorial/20260927_v1`.

The 60 training sets fit 360 logistic heads. Each pair of recipes shares its
electronic/rock heads exactly; only pop/ambient differ. Four full-budget cells
separate head and rule changes. Partial budgets use fixed original rules and
report proper probability scores as the main learning-curve measures. No result
from this study alone is new-song confirmation or release authorization.
