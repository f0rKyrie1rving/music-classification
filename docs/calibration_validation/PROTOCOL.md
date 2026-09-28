# Fixed-model validation on new, documented-project-disjoint artists

This phase follows exploratory work on previously used development data. It tests
transfer of a fixed probability correction to new project track and artist IDs.
It is not encoder retraining, a new classifier search, or an application-model update.
The protocol is recorded locally before new audio predictions; it is not externally
preregistered. Candidate metadata and its availability were inspected first.

## Fixed system and comparison

Use the originally designated primary split, seed 20260926, from
`outputs/calibration/20260926_v2`: the classifier fitted on 733 tracks and sigmoid
calibrators fitted on 232 different tracks. Use its already saved conservative policy
from `outputs/calibration_conservative/20260926_v1`: lambda [0, 0, .25, 0] for
electronic, pop, ambient, rock. Reconstruct saved scaler/head parameters and verify
against old saved calibration/evaluation logits before computing any new predictions.
Do not choose the best of the five historical models, refit any component, retune
lambda, change the ontology, or attach these calibrators to the app's final classifier.

The three methods are raw, full sigmoid, and conservative sigmoid. Primary contrast:
conservative minus raw macro binary Brier. Full sigmoid is a fixed secondary comparator.
The new scores must not drive any changes to this system or the sample. Null and harmful
outcomes are retained. Any subsequent model development consumes this validation set;
it cannot remain an untouched test for another iteration.

## Sampling frame, not the entire music catalogue

Official MTG-Jamendo genre metadata at Git commit
`cafd8e20c265ed84f1e61f1c875327971f43a62f`, with exact Git blobs and SHA-256 checks.
The acquisition frame is the next four numerically ordered archive shards after the
project's old 00–03 frame: **04, 05, 06, 07**. This restricted frame controls acquisition
cost; it is not a random sample of the whole Jamendo catalogue.

Exclude all 1,535 historical project track IDs and all 583 corresponding artist IDs,
including past training, validation, historical tests, learning/listening samples and
candidate pools. The project audit found no audio IDs outside this exclusion set.
Keep tracks of at least 30 seconds, with an official audio checksum and an explicit
CC BY, BY-SA, BY-NC or BY-NC-SA license. Preserve individual attribution. No new license
policy is inferred for other license types; they are outside this declared frame.

Before any model prediction, the verified metadata frame contained **394 tracks from
254 eligible artists**. Hash-sort artists with the fixed seed in config.json, take
200, then hash-sort each artist's eligible songs and take up to two. This selects
**266 tracks from 200 artists**, according to an independent metadata-only reconstruction.
No model score or label quota enters selection. Keep all-zero and multi-positive broad
targets. All eligible songs have at least one source genre annotation; the inference
target is the declared sample, not genre-unannotated tracks or a natural streaming mix.

Sample size is a bounded first validation, not a power guarantee. Two hundred artist
clusters enable an interval estimate without pretending songs from one artist are
independent. Precision and positive-label artist counts must be reported, even if
they are poor; do not acquire extra tracks in response to statistical significance.

## Acquisition and failures

Read TAR headers by exact HTTP byte ranges; do not download whole multi-GB archives.
Use the existing bounded downloader and CPU preprocessing: first 30 seconds,
22,050 Hz mono PCM16 WAV. It allows up to 1,310,720 encoded bytes per selected track,
validates exact HTTP 206 ranges/archive lengths, handles leading ID3 metadata, and
requires a finite, non-silent 30-second decode. Its fixed per-request retries are
allowed; no replacement tracks, genre quota repairs or score-based exclusions.

Retain every selected ID and record each acquisition failure. Partial MP3 prefixes
are locally hashed and range-checked, **not** described as full-file checksum verified.
Use full-file verification only when all official member bytes were actually acquired.
Record total bytes, local WAV hashes and the fraction of tracks/artists successfully
observed. Acquisition failure may bias the retained sample; evaluate the observed
subset with that qualification and disclose label/artist coverage.

Extract frozen MAEST block-7 CLS/DIST/mean features, using the original model revision,
extractor, CPU backend and package versions. Before processing new songs, verify a
known historical song reproduces its retained feature vector. Preserve feature order,
audio hashes and decode/extraction failures. Failed rows are never replaced or silently
treated as valid predictions. If any needed input cannot be verified, stop that stage.

## Evaluation

Primary metric remains the track-weighted mean of the four label Brier scores, for
continuity with the exploratory study. Artist sampling with up to two songs changes
the sample's weighting relative to a catalogue-uniform song sample; no population
prevalence or deployment claim is inferred. Report per-label Brier, binary log loss,
AP, positive tracks/artists, raw probabilities, and fixed 5/10-bin reliability plots.
Brier/log loss quantify probability quality, not pure calibration or recognition
accuracy in isolation. These monotone per-label transformations do not improve ranking.

Use 2,000 paired artist-cluster bootstrap samples, seed 2026092702, retaining all
four labels and all method predictions for each song. The raw/full-sigmoid comparisons
share exactly the same sampled artists. Report conditional, pointwise percentile
95% intervals; no multiplicity adjustment or classifier/calibrator refitting is
included. An interval crossing zero is inconclusive evidence about the mean change,
not proof that the methods are equivalent. Do not continue tuning until it excludes zero.

## Novelty boundary and external history

Newness is relative to documented **project** use and artist IDs. Alias/collaboration
overlap and MAEST pretraining overlap remain unknown. A user question about undocumented
Jamendo testing/listening outside the project was raised during preparation. Save the
response if available. Without a response, record that exposure history is unconfirmed;
do not silently assume none or claim pristine independence from all prior exposure.
New source tags are existing uploader metadata, not newly created human ground truth.

## Provenance

Preserve the excluded/candidate/selected/observed manifests, source snapshot hashes,
config/protocol, fixed model hashes, metadata Git blobs, acquisition receipts, feature
order/hashes/runtime, all full-precision predictions, paired bootstrap draws and reports.
New outputs live only in `outputs/calibration_validation/20260927_v1` and new research/
documentation directories. Previous research and application files remain read-only.

Source documentation: [official MTG-Jamendo repository](https://github.com/MTG/mtg-jamendo-dataset)
and [official downloader](https://github.com/MTG/mtg-jamendo-dataset/blob/cafd8e20c265ed84f1e61f1c875327971f43a62f/scripts/download/download.py).
Audio is used locally for non-commercial research under the retained source conditions.
