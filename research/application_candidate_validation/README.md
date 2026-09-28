# Fixed-candidate confirmation on new project songs

This stage compares the frozen candidate in `artifacts/candidates/20260927_v1/`
with the actual v1.1 application on the same newly acquired songs. It does not
fit a model, change a threshold, replace the application, or select another test
cohort after inspecting results.

**Status: completed, `go` under the prespecified confirmation rule.** All 372
selected tracks / 250 artists were acquired and extracted, with no failures or
replacements. Focus false positives fell 152 → 124 (18.42%); focus false negatives
rose 59 → 61. Micro F1 increased 0.619910 → 0.637002; macro Brier decreased
0.128989 → 0.125475 (2.72%). Coverage, point safeguards and both confidence gates
passed. Exact replay and independent reconstruction passed. The application has
not been replaced or released. See the [Chinese report](../../docs/application_candidate_validation/SUMMARY_ZH.md)
for full results and limits.

## Fixed data boundary

- Official metadata commit: `cafd8e20c265ed84f1e61f1c875327971f43a62f`.
  `config.json` records the exact Git blobs for genres, licenses and checksums.
- Source archives: `12` through `19`, fixed before acquisition.
- Historical exclusions: 2,195 track IDs and 1,008 artist IDs, from the frozen
  candidate's `future_exclusions.json`.
- Eligible frame after exclusions, licensing and duration checks: 640 songs
  from 288 artists.
- Fixed sample: **372 songs from 250 artists**, at most two songs per artist.
  Artists and then their songs are ordered by SHA-256 using
  `music-candidate-confirmation-20260928-v1`. Selection uses IDs, not labels or
  predictions. `selected_manifest.json` is fixed before any new archive request.
- The entire eligible frame is retired from future fresh evaluation, including
  unselected songs and failed acquisitions. The resulting exclusion boundary is
  **2,835 track IDs and 1,296 artist IDs**.

“New” means absent from documented project exposure. This is still the same
dataset source. Artist aliases and upstream MAEST pretraining overlap are unknown.
The official genre tags are proxy labels; no human listening or relabeling occurs.

## Commands and execution order

Run from the repository root with the existing `.venv-improve` environment. The
commands below document the one frozen run; do not run the preparation, freeze
or evaluation commands again against completed outputs.

These are commands for this existing research workspace, not a clean-clone
one-command setup. They require the frozen candidate bundle, the current v1.1
application artifacts, the local MAEST checkpoint, historical source metadata,
the development control audio/cache, and the prior verification records named in
`pipeline.BASE_INPUTS` and `freeze.json`. No model or cache is silently rebuilt.

```sh
# Run from the music-classification repository root.
candidate_validation_out="outputs/application_candidate_validation/20260928_v1"

# Local metadata only: audit the whole frame and fix selected IDs.
.venv-improve/bin/python -m research.application_candidate_validation.data --out "$candidate_validation_out"

# Freeze source, model, runtime, source metadata and selected IDs before networking.
.venv-improve/bin/python -m research.application_candidate_validation.pipeline freeze --out "$candidate_validation_out"

# Network stages, using the fixed sample and acquisition policy.
.venv-improve/bin/python -m research.application_candidate_validation.acquire index --out "$candidate_validation_out"
.venv-improve/bin/python -m research.application_candidate_validation.acquire resolve --out "$candidate_validation_out"
.venv-improve/bin/python -m research.application_candidate_validation.acquire download --out "$candidate_validation_out"

# Local CPU extraction, one fixed model comparison, and exact result replay.
.venv-improve/bin/python -m research.application_candidate_validation.pipeline features --out "$candidate_validation_out"
.venv-improve/bin/python -m research.application_candidate_validation.pipeline evaluate --out "$candidate_validation_out"
.venv-improve/bin/python -m research.application_candidate_validation.pipeline verify --out "$candidate_validation_out"

# Independent reconstruction of provenance, predictions, metrics and uncertainty.
.venv-improve/bin/python docs/application_candidate_validation/independent_verify.py --out "$candidate_validation_out"
```

The network commands need an execution context permitted to reach
`https://cdn.freesound.org`. A restricted execution environment produced DNS
failures on a preflight request; archive indexing and downloading therefore
require an environment with working network access. Do not
start the actual sample download from a context known to block network access:
recorded terminal failures are part of the fixed run and cannot simply be erased.

To inspect integrity without acquiring data or recomputing model scores:

```sh
.venv-improve/bin/python -m research.application_candidate_validation.pipeline verify_frozen --out "$candidate_validation_out"
```

The isolated acquisition tests make no network requests:

```sh
.venv-improve/bin/python -m unittest discover -s tests -p 'test_candidate_validation_data.py' -v
```

## Acquisition, checkpoints and recovery

Indexing follows only 512-byte TAR headers; it does not download whole archives.
Every verified header is saved under `index_headers/<archive>/<offset>.bin`, with
hashes in the archive index and per-request audit records. Each completed header
position is persisted, so an allowed indexing resume starts at the last saved
position. Each archive permits at most **three indexing invocations** in total.
Completed indices are verified and reused without issuing further requests.

Each HTTP range request has at most three attempts, an 8-second connection
timeout and a 25-second request timeout, with fixed 1/2/3-second retry waits.
The response must contain exactly the requested bytes with HTTP 206 and the
matching `Content-Range`. All attempts are recorded in `network_requests/`.
Indexing and downloading each use four workers.

`resolve` adds verified archive offsets to a separate `resolved_manifest.json`.
It binds the selection, freeze and all archive-index hashes; the original selected
manifest remains byte-for-byte unchanged.

Each song has one durable `download_checkpoints/<track_id>.json` record. A completed
success is reused only after its receipt and decoded-audio hash are checked. A
terminal failure is never retried. If a process ends after recording `started`
without recording completion, that song becomes a documented failure on resume.
There are no replacement songs. Per-range retries within that one song acquisition
are the only automatic download retries.

The downloader fetches at most 1,310,720 encoded bytes per song and decodes the
first 30 seconds to mono 22,050-Hz PCM16 WAV. It rejects insufficient, silent or
nonfinite audio. Partial prefixes are not described as full-file checksum verified.
Licenses, byte ranges, encoded hashes and decoded WAV hashes are preserved.

Process lock files prevent concurrent operations. If a lock remains after an
interruption, first inspect whether its process is still running and inspect the
saved checkpoints. Do not delete checkpoints, reset invocation counts or remove
failure records to obtain more attempts. Only a confirmed stale process marker may
be cleared before the permitted resume. Once `download_status.json` exists,
acquisition is complete and the download command refuses another run.

Feature extraction begins only after every selected track has a recorded
acquisition outcome. It verifies a historical control vector before extracting the
new cohort, then preserves each successful or failed feature checkpoint. Both
models receive exactly the same usable rows in fixed selection order. Predictions
begin only after acquisition, extraction and missing-data accounting are complete.

## Assessment and records

The complete prespecified rules are in [protocol.md](protocol.md) and
[config.json](config.json). The primary comparison concerns pop and ambient under
the fixed `f1` policy. It requires at least a 10% reduction in their combined false
positive cases, with fixed recall, F1 and probability-error safeguards. Electronic
and rock provide exact unchanged-label controls. The other fixed application
policy and all four labels are also reported.

A formal `go` additionally requires sufficient track/artist and positive/negative
label coverage, plus both prespecified confidence conditions: the upper 95%
interval bound is below zero for the change in focus false-positive cases per
track and for the change in macro Brier error. Intervals use 2,000 paired bootstrap
draws clustered by artist, with the seed fixed before acquisition. The two models
are compared within the same bootstrap draws. Failed coverage, failed performance
conditions and inconclusive intervals remain visible; no metric, cohort, seed or
threshold is selected afterward.

The run directory retains:

- `freeze.json`, `source_snapshot/`, `model_snapshot/`, official source metadata,
  exclusions, frame, fixed selection and resolved archive locations;
- complete raw TAR headers, request attempts, per-song acquisition checkpoints,
  audio receipts and final download outcomes;
- extraction control, feature checkpoints, usable-row manifest and feature hashes;
- aligned predictions, fixed bootstrap draws, metrics, intervals, coverage and
  assessment when evaluation completes;
- exact replay and independent verification reports when their checks complete.

The application is not replaced automatically, even if the candidate passes.
An uncertain or failed result ends this candidate's single confirmation run and
is reported without modifying the frozen candidate or this sample.

For this completed run, the 95% interval for relative focus FP change is
[-26.90%, -9.70%]. The observed reduction exceeds 10%, but the interval does not
establish that the reduction is at least 10%. The formal confidence gates instead
test whether the entire focus-FP-rate and macro-Brier difference intervals are
below zero. Individual metric intervals are conditional and generally descriptive;
this is not evidence about arbitrary music, independent datasets or human-adjudicated
genre correctness. The outcome supports a separate application-integration stage.

This README documents operation and status. It is outside the frozen executable
source inventory; the protocol and configuration remain frozen authoritative
inputs. Existing research outputs and prior evaluation claims are preserved.
