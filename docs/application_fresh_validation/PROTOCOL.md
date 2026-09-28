# Application v1.1: fixed new-project-song acceptance check

This local protocol is fixed before acquiring selected audio, extracting features,
or predicting on this cohort. It is not external preregistration. Candidate
metadata and availability were inspected first. Do not refit the model, select a
different correction, alter thresholds, replace failures, or change statistical
criteria after seeing any new scores. Keep unfavorable outcomes.

## Fixed application and question

Compare the exact v1.0 final 1,206-track classifier with the v1.1 application at
Git commit 931c58b. Use its unchanged frozen MAEST encoder and linear heads.
The v1.1 artifact subtracts 1.3718462253725046 from ambient log-odds, based only
on its original 244 positive / 962 negative training counts. Other labels are
unchanged. Tags are still selected using original unrounded scores/thresholds.
This is a test of probability error on new project songs, not a new classifier,
new algorithm, or promised improvement in recognition accuracy.

## Newness and fixed sampling frame

Use the verified official MTG-Jamendo genre/license/checksum metadata at commit
cafd8e20c265ed84f1e61f1c875327971f43a62f. Restrict acquisition to the next four
archive shards 08, 09, 10, 11. Exclude every track and artist from the original
project manifests and candidate pools, including all 394 previously inspected
04–07 candidates, rather than just the 266 songs already scored. Record hashes
of every history source and audit local audio/cache IDs.

Require at least 30 seconds and an official full-file reference checksum; allow
only explicit CC BY, BY-SA, BY-NC and BY-NC-SA licenses, with verbatim attribution.
This uses the same bounded local research acquisition as prior work; no raw
audio or third-party encoder weights are published by this task.

The initial resource target was 200 artists. The metadata-only audit found
266 eligible songs from 171 artists. Before any new audio acquisition or
prediction, fix the sample to **all 171 eligible artists**, selecting up to two
songs per artist by SHA-256 ordering with the seed in config.json. The amendment
is preserved in metadata_amendment.json. Do not expand the frame to reach 200.
Artist selection is therefore a census of eligible artists in these four shards,
not a random 200-artist sample of the whole catalogue. No genre quotas or model
scores enter selection. Preserve songs with zero or multiple target labels.

Newness is only relative to documented project track/artist IDs. This remains
the same source dataset, with uploader tags rather than new human ground truth.
Unrecorded external exposure, artist aliases/collaborations, and encoder
pretraining overlap cannot be ruled out. Do not call this an independent dataset
or general proof about all user music.

## Acquisition and runtime

Index exact TAR headers using bounded HTTP byte ranges. Use the existing
range-checked downloader, capped at 1,310,720 encoded bytes per track, including
the established fixed per-request retries. Decode exactly the first 30 seconds
to 22,050 Hz mono PCM16 without gain normalization; require finite non-silent
audio. Partial prefixes have local segment/WAV hashes, not full-file verification
unless the entire member was actually fetched and matched the official hash.

Record every selected ID and all technical download/extraction failures, with
no replacements or outcome-based exclusions. Before new songs, reproduce an old
MAEST feature vector on CPU with the original dependency versions and maximum
absolute error <= 1e-5. Hash features, audio, source code, models, and receipts.
Freeze the sample, protocol, code and dependencies before downloading songs.
Resumption must verify the freeze and completed-row hashes.
Persist a per-track download checkpoint before and after each attempt. Preserve
completed failures without further attempts; a request interrupted with unknown
completion is recorded as a technical failure, not silently retried. An exclusive
process marker prevents simultaneous acquisition runs; a stale marker requires
inspection before resumption.

## Evaluation and decision rule

Primary: track-weighted mean of the four binary Brier scores (lower is better).
Secondary: mean binary log loss, with probabilities clipped to [1e-12,1-1e-12]
for this calculation. Report per-label Brier, log loss, average precision, mean
estimate, label prevalence, positive track and artist counts. These proper
scoring rules measure overall probability quality, not pure calibration alone.

Compare both existing threshold policies' precision, recall, F1, false positives,
false negatives, and output coverage. Require exact agreement of old/new selected
tags and of the new raw path with the historical predictor, including boundary
handling and disabled 1.01 thresholds. Within-label ranking is unchanged by the
monotone correction; cross-label score ordering may change.

Use 2,000 paired artist-cluster bootstrap draws, seed 2026092802, with identical
draws for both methods. Report pointwise percentile 95% intervals of corrected
minus raw Brier and log loss. These are conditional artist-resampling sensitivity
intervals / superpopulation approximations, not design-based confidence intervals
for the finite census of 171 artists. They do not include refitting, selection,
annotation uncertainty, or multiplicity adjustment. Sample size is not a power
guarantee. An interval crossing zero is inconclusive, not evidence of equivalence.

Classify evidence in this order, without changing the application from these results:

1. Integrity failure: stop and report any cache/model/API/decision inconsistency.
2. Harmful on the observed subset: Brier or log-loss difference interval lower
   endpoint > 0; always disclose any low coverage alongside this signal.
3. Incomplete evidence: fewer than 80% of selected tracks OR selected artists
   observed; do not call the validation successful.
4. Go: both point estimates decrease, Brier difference upper endpoint < 0,
   sufficient observed coverage and no changed decisions.
5. Promising: both point estimates decrease but Brier interval includes zero.
6. Otherwise inconclusive.

The gate concerns this declared application probability comparison only; a go
does not certify a Windows build or justify broad deployment/publication claims.
Save all predictions and failure records. Inspect every label's errors and a
deterministic list of most severe observed false positives/negatives, explicitly
as post-evaluation diagnosis. Source-label disagreements are not automatically
human-confirmed mistakes. These songs become exposed after this check and cannot
be reused as untouched validation for later model changes.
