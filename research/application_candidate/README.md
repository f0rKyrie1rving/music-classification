# Fixed application candidate

This stage creates a separate runnable candidate after the exploratory
[factorial diagnosis](../application_factorial/README.md). It changes pop and
ambient only, preserving the actual installed application's electronic/rock
parameters and decisions. It does not release or install the candidate.

The [protocol](protocol.md) and [config](config.json) fix four artist/cohort OOF
folds, one threshold-selection procedure and two final full-development fits.
There are 18 fitted heads, with no hyperparameter or seed search. OOF selection
scores are not held-out performance evidence.

```sh
.venv-improve/bin/python -m unittest discover -s tests -p 'test_candidate_prediction.py' -v
.venv-improve/bin/python -m unittest discover -s tests -p 'test_application_candidate_build.py' -v
.venv-improve/bin/python -m research.application_candidate.build freeze
.venv-improve/bin/python -m research.application_candidate.build train
.venv-improve/bin/python -m research.application_candidate.build verify
.venv-improve/bin/python -m research.application_candidate.build smoke
.venv-improve/bin/python docs/application_candidate/independent_verify.py
```

Freeze and train are one-time commands; verify and smoke do not fit models.
Do not delete failed/incomplete fit receipts to silently repeat a fit. All source
snapshots, data roles, training subsets, OOF support models and selection audits
are retained in `outputs/application_candidate/20260927_v1`.

The standalone candidate bundle is `artifacts/candidates/20260927_v1`. Its strict
loader verifies every bundled receipt and the non-executable parameter archive.
Run the candidate and actual application on one shared audio feature vector:

```sh
.venv-improve/bin/python candidate_predict.py data/training_audio/track_0001100.wav --compare-baseline
```

Use `--policy precision_target` to preserve the installed selective policy's tag
decisions: copied electronic/rock thresholds and disabled pop/ambient labels.
The focus scores still reflect the newly fitted heads. No old ambient weight
offset is applied to the new unweighted classifier.

The candidate has no new-song validation result. Freeze a separate acquisition
and acceptance protocol before comparing it with the actual installed v1.1 app
on new songs, and preserve this candidate's artifact hashes through that test.
