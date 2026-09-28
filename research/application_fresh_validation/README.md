# Fixed application validation on new project songs

This is a single, frozen comparison of the released 1,206-track application's
original probabilities and its v1.1 class-weight correction. It does not train
or tune anything. Read `protocol.md` before interpreting the results.

The fixed run lives in `outputs/application_fresh_validation/20260927_v1`.
Its source snapshot, model hashes, source metadata Git blobs, 232 selected
track IDs, 171 artist IDs, and 1,929 historical exclusion IDs were recorded
before new audio acquisition. All 394 earlier validation candidates are part
of the exclusion, not just the previously scored songs.

Modules:

- `data.py`: metadata audit, byte-range archive index, fixed selection, bounded
  download with persistent success/failure checkpoints.
- `pipeline.py`: source/model/runtime freeze, original-feature control, verified
  and resumable per-song MAEST feature extraction.
- `evaluate.py`: the one fixed application comparison, paired artist bootstrap,
  unchanged-decision checks, and prespecified evidence classification.
- `report.py`: descriptive Chinese report and source-label disagreement examples.

Synthetic verification (no network or new song inference):

```sh
.venv-improve/bin/python -m unittest research.application_fresh_validation.test_data research.application_fresh_validation.test_pipeline research.application_fresh_validation.test_evaluate -v
```

Verify the frozen local run:

```sh
.venv-improve/bin/python -m research.application_fresh_validation.pipeline verify_frozen --out outputs/application_fresh_validation/20260927_v1
```

An independently written checker lives at
`outputs/application_fresh_validation/independent_verify.py`. Once evaluation
has completed, it can reconstruct metadata, all predictions, both policies,
all metrics, and every bootstrap draw from the local evidence:

```sh
.venv-improve/bin/python -B outputs/application_fresh_validation/independent_verify.py --out outputs/application_fresh_validation/20260927_v1
```

These full checks require the existing local metadata, feature caches, audio
receipts and encoder files. They are not a promise that a clean source checkout
contains licensed audio or all historical research caches. Audio and third-party
encoder weights remain local. The generated report includes numerical results
and track attribution without distributing audio.

Do not rerun acquisition to erase failures, overwrite this run, or reuse the
same songs as untouched validation of a subsequently changed model. Statistical
success is separate from Windows build/install verification and publication.
