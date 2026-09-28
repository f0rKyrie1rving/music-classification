# Fixed-model validation on new project artists

This phase tests the already selected primary research model and its probability
correction on additional MTG-Jamendo songs. It does not retrain the encoder, select a
new classifier, or update the app model. Read [protocol.md](protocol.md) first.

The documented historical pool contains 1,535 songs from 583 artists. All are excluded.
The additional sampling frame uses official archive shards 04–07, exact upstream
metadata blobs, permitted source-license categories, and existing genre annotations.
It contains 394 candidate songs from 254 artists. Fixed artist/track hashing selects
200 artists and at most two songs per artist: 266 songs total. No prediction or label
quota is used in selection. Generalization is limited to this explicit sample/frame;
unrecorded prior exposure, artist aliases, and encoder pretraining overlap remain
unverified. Source audio remains local and is not included in the report.

## Reproduction

Run at repository root. Use a new output directory for another run; never overwrite
past frozen inputs or results. The metadata files below must be downloaded to its
`source/` directory first from the pinned upstream Git commit. `data audit` verifies
their Git blob identities, including exact bytes.

| Local source name | Upstream path at commit `cafd8e20c265ed84f1e61f1c875327971f43a62f` |
| --- | --- |
| `genres.tsv` | `data/autotagging_genre.tsv` |
| `licenses.txt` | `audio_licenses.txt` |
| `checksums.txt` | `data/download/raw_30s_audio-low_sha256_tracks.txt` |

```sh
.venv-improve/bin/python -m unittest research.calibration_validation.test_data research.calibration_validation.test_evaluate -v
.venv/bin/python -m research.calibration_validation.data audit --out outputs/calibration_validation/20260927_v1
.venv/bin/python -m research.calibration_validation.data index --out outputs/calibration_validation/20260927_v1
.venv/bin/python -m research.calibration_validation.data sample --out outputs/calibration_validation/20260927_v1
.venv-improve/bin/python -m research.calibration_validation.evaluate verify-model --out outputs/calibration_validation/20260927_v1
.venv-improve/bin/python -m research.calibration_validation.pipeline freeze --out outputs/calibration_validation/20260927_v1
.venv/bin/python -m research.calibration_validation.data download --out outputs/calibration_validation/20260927_v1
.venv-improve/bin/python -m research.calibration_validation.pipeline features --out outputs/calibration_validation/20260927_v1
.venv-improve/bin/python -m research.calibration_validation.evaluate score --out outputs/calibration_validation/20260927_v1
.venv-improve/bin/python -m research.calibration_validation.verify_results --out outputs/calibration_validation/20260927_v1
.venv-improve/bin/python -m research.calibration_validation.report --out outputs/calibration_validation/20260927_v1 --destination docs/calibration_validation
```

Before `verify-model`, create `model_reference.json` with `base_run` set to
`outputs/calibration/20260926_v2`, `conservative_run` set to
`outputs/calibration_conservative/20260926_v1`, and `primary_seed` set to `20260926`.
Before freezing, record external-use history or its unresolved status in
`exposure_history.json`; an unanswered question is not evidence of no outside exposure.

The index stage resumes its own checkpoints and only reads bounded TAR headers. The
download stage reuses the original reviewed downloader with all paths redirected to
this run. Each HTTP response must match the requested byte range; partial MP3 files
are not advertised as official whole-file-hash verified. Every acquisition/extraction
failure is recorded; no replacement songs are allowed. Once a final stage status or
final features exist, it refuses to overwrite them.

The feature stage uses the existing `.venv-improve` audio/ML dependencies and frozen
CPU encoder. It first reproduces a historical feature vector. The inference stage
uses saved numerical coefficients and calibrator parameters, without any fitting.
The raw, full sigmoid, and conservative predictions use exactly the same observed
tracks; bootstrap comparisons use matching artist samples.

## Artifacts

The run directory retains official metadata checks, exclusions/candidates/selected and
observed manifests, download range receipts, feature provenance, fixed model verification,
frozen sources, per-song predictions, metrics, and paired bootstrap draws. Full reports
retain unhelpful as well as helpful results. Unit tests and independent numerical checks
are separate. This validation data becomes part of project history after evaluation;
it cannot be called untouched evidence for subsequent method changes.

Data source: [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset).
