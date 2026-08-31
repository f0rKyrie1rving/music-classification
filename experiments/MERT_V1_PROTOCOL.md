# MERT-v1 representation experiment

Frozen after the expanded MERT-v0 development result was known and before
MERT-v1 weights, features, or downstream outcomes were inspected.

## Single changed factor

Replace MERT-v0-public with the official `m-a-p/MERT-v1-95M` encoder. The
official model card reports the same 95M scale and 12 × 768 transformer geometry,
but 24 kHz input and roughly 20,000 pretraining hours rather than v0-public's
16 kHz and 900 hours. It states that v1 models outperform earlier versions and
generalize better to downstream tasks. Pin an exact repository revision, review
local custom code, verify every file and use restricted checkpoint loading.
License: CC BY-NC 4.0 according to the official model card.

Official source: https://huggingface.co/m-a-p/MERT-v1-95M

## Unchanged development boundary

- Use the same 903 training and 303 development-validation tracks, frozen broad
  ontology, five artist-grouped folds, and 329 excluded test IDs in
  `experiments/expanded_model_plan.json`.
- Do not download, encode, inspect, fit, or score any test audio.
- Resample each verified development WAV to the encoder's declared 24 kHz,
  split its first 30 seconds into six 5-second chunks, mean-pool time, and store
  all 13 returned 768-wide layer representations.
- Search the same 13 individual layers plus transformer-layer mean and the same
  six logistic-regression settings. Selection remains per label by mean grouped
  training-CV average precision. Reuse the existing fixed folds exactly.
- Choose the same F1 and precision-target policies from training OOF scores and
  apply them without retuning to the same 303 development-validation tracks.

## Decision rule

Compare v1 with v0 on the same broad-label development protocol. Do not promote
v1 merely because one metric rises. The existing provisional goal still
requires micro precision >=0.80, every label precision >=0.80 and recall >=0.30,
at least 10 validation predictions per label, and output coverage >=0.50.

If the goal fails, keep the holdout sealed and report which labels improved.
The next possible route is a dedicated music-style model such as MTG's
Discogs-trained MAEST, but it requires a separate dependency, license, taxonomy,
and evaluation protocol and is not silently substituted in this experiment.
