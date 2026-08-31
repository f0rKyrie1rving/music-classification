# Expanded broad-genre model protocol

Frozen before extracting expanded MERT features or fitting the expanded model.
The earlier 300/90 MERT results and exploratory error audit are already known.

## Product task and target ontology

The user-facing task is four broad, multi-label genres rather than exact string
matching against uploader tags.  A conservative ontology is fixed here before
the new holdout is used:

- `electronic`: electronic, electronica, edm, house, deephouse, techno, trance,
  idm, dubstep, drumnbass, breakbeat, triphop, downtempo, electropop, synthpop.
- `pop`: pop, poprock, popfolk, instrumentalpop, electropop, synthpop.
- `ambient`: ambient, darkambient, atmospheric, chillout, newage.
- `rock`: rock, alternativerock, hardrock, instrumentalrock, postrock, punkrock,
  classicrock, bluesrock, rocknroll, grunge, poprock.

An audio item is positive for a broad genre when at least one of its source tags
is in that genre's fixed set.  A tag may map to two broad genres, as required by
the multi-label task.  Ambiguous terms such as `alternative`, `dance`, `indie`,
`metal`, `easylistening`, and `lounge` are deliberately not inferred.  No model
score or listening judgement changes this mapping.  The earlier exact-tag task
remains reported as a historical baseline; its metrics are not directly
comparable to the new target definition.

## Development data boundary

- Training: all 903 `train` rows in `data/expanded_manifest.json`.
- Development validation: all 303 `development_validation` rows.
- Historical test: 90 rows excluded because earlier results were observed.
- New holdout: 239 preselected rows.  Do not download, encode, inspect, fit, or
  score their audio during model development.
- All folds are grouped by artist.  The scaler and classifier are fitted inside
  each training fold; no validation or test features enter those fits.

## Frozen representation search

Use the reviewed, pinned MERT-v0-public checkpoint.  For each 30-second WAV:

1. resample to 16 kHz and split into six consecutive 5-second chunks;
2. for each chunk, mean-pool time for all 13 returned representations (embedding
   output 0 and transformer outputs 1–12);
3. average the six chunks, storing a 13 × 768 matrix.

The candidate representations are each individual layer 0–12 and the mean of
transformer layers 1–12.  The upstream model card says layers may perform
differently by downstream task.  Layer selection occurs only through training
out-of-fold predictions.  No crop, layer, or pooling choice is added after
development outcomes are seen.  The encoder is frozen; audio never leaves the
machine.

## Frozen classifier search

Use five shuffled GroupKFold folds with seed 20260830.  Each genre selects its
own representation and logistic-regression head by mean fold average precision:

- StandardScaler followed by logistic regression;
- C in {0.001, 0.01, 0.1};
- class_weight in {None, balanced};
- max_iter 3000 and random_state 2026.

Exact ties use the earlier candidate in the declared order.  Per-label choices
are allowed because the model card explicitly treats layer usefulness as
task-dependent.  Selection estimates are optimistic after searching candidates,
so the 303-track development validation is reported separately.

## Operating policies and stop rule

Thresholds use only the selected training out-of-fold scores.  Report:

- an F1 policy on thresholds 0.05–0.95 in steps of 0.025;
- a precision policy requiring at least 0.80 precision, 0.30 recall, and 20
  emitted out-of-fold predictions per label, then maximizing recall, precision,
  and proximity to 0.5 in that order. Unsupported labels are suppressed.

Report TP, FP, FN, precision, recall, F1, AP, micro/macro summaries, and track
output coverage.  The provisional development goal requires micro precision at
least 0.80, every label precision at least 0.80 and recall at least 0.30, at
least 10 validation predictions per label, and at least 0.50 track coverage.
If it fails, do not open the new holdout; diagnose or predeclare another
development round.  If it passes, freeze the model bundle and inference code
before acquiring and evaluating the holdout once.  Reaching 0.80 is a target,
not a promised outcome or a claim about arbitrary user uploads.
