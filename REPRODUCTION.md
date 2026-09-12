# Reproduction scope

Updated 2026-09-12. Historical baseline: commit `68b51d5ba8dd9d786e8a0fa27b5e34bacd2ccb7c`.
`experiments/review_baseline.json` records the original hashes of protected
data, models, plans and pipeline inputs before these review fixes. Original
manifests, human answers, frozen plans, model parameters and scores are retained.

## Check the published results

Install `requirements-evaluation.txt`, then run `python evaluate_release.py`.
This only needs the public repository. It recalculates both policies' complete
metrics, the original track-bootstrap intervals and all listening counts, and
checks them against the original JSON results. It also computes supplementary
artist-cluster intervals and the two learning-sample exclusions.
Optional `--output outputs/review_audit.json` saves the full audit;
`--markdown EVALUATION_REVIEW.md` regenerates its public report.

These checks are arithmetic verification of the released score table. They
cannot establish that a listener was uninfluenced by presentation order or
independently prove when an experiment was performed.

## Reproduce the fixed final model

`reproduce_final.py` provides a separate path for computational reproduction of
the already-selected final system. It does not rerun the historical MFCC/MERT/
MAEST candidate searches. Use the pinned research environment in the README.

1. `python reproduce_final.py check` verifies the published input snapshots,
   archived manifests/audits, frozen plans and released scores.
2. `python reproduce_final.py download --workers 8` obtains the 1,206 fit and
   239 holdout excerpts through the original bounded downloader. It checks
   each WAV against the archived audio hash, leaves matching files in place,
   and rejects existing mismatches. It never downloads the 90 historical test
   tracks. The documented <=0.1-second short-source repair remains bounded.
3. `python prepare_maest_hf.py` retrieves and verifies the pinned MAEST files.
4. `python reproduce_final.py extract --device cpu` extracts the fixed MAEST
   vectors. Partial progress is resumable; completed caches must match IDs,
   audio hashes, model/source identity, device, versions and feature checksum.
5. `python reproduce_final.py run --device cpu` fits the frozen four heads and
   thresholds and reports score differences and decision mismatches against
   the archived 239 predictions. Existing replay results are never overwritten.

New features and results live under `outputs/reproduction/`. Feature provenance
and matrix hashes are stored separately from elapsed time and resume counts.
Original final-plan hashes are not refreshed. Numerical differences across
platforms are reported rather than silently replacing the original scores;
reconstructed audio must still match the archived WAV bytes. The environment
was recorded on Apple Silicon macOS; other platforms have not been validated.
Audio/model downloads depend on upstream availability and their usage terms.

The exact retained `training_pool.json` and four completed archive indexes in
`data/source/` are now explicitly included, so the former missing-input blocker
is removed. All other files in that directory remain ignored. These are frozen
metadata/index snapshots, not a new sample or newly generated source metadata.
They retain MTG-Jamendo source references and per-track attribution. Metadata
is under CC BY-NC-SA 4.0; see [data/DATASET.md](data/DATASET.md). Each recording
retains its individual license; publishing indexes does not redistribute audio.

## Retained local feature replay

Where the original feature arrays are available, run:

```bash
python reproduce_final.py run --retained-features
```

This validates the original matrix hashes and stable development provenance,
refits the fixed heads, and compares every score/decision. It does not require
re-extraction, and is not evidence of a complete fresh download/extraction run.
Only historical extraction timing/resume counts are excluded from metadata
identity; IDs, audio hashes, versions, source hashes and matrix bytes are checked.

## Historical workflows and review limits

Historical runners and their frozen plans remain available at the baseline
commit. Some require additional retained local research inputs, including
historical encoder receipts, caches, and runtime-dependent byte-level records.
They are not advertised as a clean-clone reproduction of the whole research
history. The new final-model replay avoids those historical dependencies.

The completed development listening sheet is now protected by exclusive file
creation and an early refusal in `diagnose_maest_hf.py`. An existing sheet,
including a blank one, is left byte-for-byte intact. This diagnostic script is
not a hashed input of the original final plan, so its repair leaves that plan's
integrity checks intact. The final listening questions/answers retain their
original order; future audits should globally shuffle newly sampled questions
with a recorded seed before listening.

## Verification performed for this revision

- All 58 tests passed, including protection of the real completed listening
  sheet's bytes, interrupted extraction/resume with synthetic vectors, refusal
  of corrupted caches, and keeping all tracks of a sampled artist together.
- A clean export containing only tracked and newly included repository files
  successfully recalculated both policies' metrics, original intervals and
  listening counts, then passed `reproduce_final.py check`. It had no retained
  audio collection, model cache, or original `outputs/`. The existing local
  Python environment was used; dependency installation was not re-tested.
- All 1,445 retained fit/holdout WAVs passed the new acquisition command's
  archived-hash checks. Matching files were reused; no fresh network acquisition
  of the entire collection was performed.
- Both final policies were replayed by refitting their fixed heads from the
  retained feature matrices. Across all 239 tracks, maximum absolute score
  difference was 0.0 and both policies had zero changed label decisions.
- Fresh CPU extraction of `track_0001100` (fit) and `track_0012301` (holdout)
  matched their archived float32 feature vectors exactly. Full fresh extraction
  of all 1,445 tracks was not repeated in this maintenance pass.
- Historical protected-file hashes passed, document links were checked, and
  the updated results figure was visually inspected. Its scores are unchanged.

These checks support the repaired workflows without claiming an end-to-end
fresh-machine download, installation and extraction run or a new performance test.
