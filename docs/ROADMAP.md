# Improvement roadmap

Updated 2026-08-31. This is a work list, not a frozen experimental protocol.
The existing MAEST release artifact, source-label results, and single-reviewer
audit remain the historical baseline. A pending item does not imply that the
work has been performed or that it will improve the model.

## First pass: documentation and report presentation

- [x] Restore the four missing historical report figures using repository
  assets, and update their generators so rerunning a report preserves the links.
- [x] Separate the README's 300/90 exact-tag and 903/303 broad-label comparisons;
  identify the selected MFCC head and distinguish validation from historical test.
- [x] Add a retrospective design rationale with source references and limits.
- [x] Verify report metrics, figure rendering, local links, and unchanged model,
  data, training code, and frozen protocol files.

Completed locally on 2026-08-31. All four generators ran against copies of saved
inputs in a temporary directory, with their original integrity checks retained.
Metric tables were unchanged and rendered figures matched the archived images
pixel for pixel. Local documentation links were checked and 97 protected files
retained their hashes. No model was retrained and no new predictions were made.

## Repository maintenance

- [ ] Tag or release a verified baseline before reorganizing code. Retain its
  environment and reproduction instructions; do not rewrite experiment history.
- [ ] Run the existing lightweight tests in GitHub Actions without requiring
  full audio acquisition or an encoder download on every change.
- [ ] Introduce `pyproject.toml` and clearer package/script/report boundaries
  incrementally. Check imports, CLI commands, paths, and frozen source hashes.
- [ ] Add relevant repository topics and concise release notes.

## Calibration and selective-output study

- [ ] Write a separate protocol before fitting calibrators: define the target
  distribution, primary metric, candidate methods, artist-grouped fitting and
  evaluation splits, and handling of failed or inconclusive results.
- [ ] Reuse the existing grouped-CV/OOF machinery; do not present it as new work.
  Keep calibration fitting separate from evaluation, account for prior head
  selection, and retain an uncalibrated baseline.
- [ ] Compare per-label probability quality with Brier score, reliability
  diagrams including bin counts, and a declared ECE binning rule. Include a
  training-prevalence constant baseline and examine ambient's class weighting.
- [ ] Start with a bounded per-label sigmoid-calibration comparison. Preserve
  the multi-label semantics; probabilities do not have to sum to one. Do not
  claim calibration improves ranking or solves incomplete source tags.
- [ ] Add precision/recall/output-coverage curves with emitted-label counts.
  Define risk and the unit of coverage before adding risk-coverage plots;
  distinguish no applicable target label from explicit abstention.

## Statistical uncertainty and independent evaluation

- [ ] Add artist-cluster bootstrap as a supplementary analysis: sample artists
  with replacement and retain their tracks together. Declare whether metrics
  weight tracks or artists, the seed, number of replicates, interval method,
  and treatment of replicates with no positives or no predictions for a label.
- [ ] When comparing two methods on the same evaluation tracks, use the same
  resampled artist groups to estimate paired metric differences. Keep the
  original track-bootstrap results unchanged and label old-holdout additions
  as post-hoc analyses, not new independent tests.
- [ ] If making a new independent v2 performance claim, freeze the candidate
  and a new artist-disjoint evaluation plan before scoring. Exclude previously
  inspected artists, evaluate v1 and v2 on the same new tracks, and report
  results even if the improvement is absent. Documentation work alone does
  not require another holdout.

## Outside the current scope

No additional independent listeners are planned. Keep the existing
single-reviewer limitations visible; do not substitute repeated self-review,
AI-generated judgments, or invented majority-vote labels for independent
annotation. No new encoder, layer/pooling search, relabeling, or automated
experiment run is included in the first documentation pass.
