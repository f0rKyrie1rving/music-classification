# Bounded performance improvement — 2026-08-28

Written before this round's cross-validation fits or pretrained feature results.
Earlier baseline and 26-vs-46 development results are already known.

## Scope and budget

- Keep the four original genre labels and the first 30 seconds of audio.
- The user's total learning/operating budget is now 20 hours, including prior
  work. Time already spent has not been measured; this is not 20 extra hours.
- Allow longer, readable core code (rough planning range 400–700 lines), not
  additional UI, paid APIs, end-to-end neural training, or an open-ended search.
- One frozen pretrained candidate: MERT-v0-public, if its weights, license,
  source code and local runtime can be checked. No fine-tuning of the encoder.
  It was trained on Music4All and part of FMA, according to its model card;
  overlap with Jamendo has NOT been ruled out. Pin the source revision and
  checksums before extraction. Do not silently switch to a different model.

## Development evaluation

1. Preserve the original manifest, baseline model, source code, caches and
   results. Do not extract or predict the old 90 test tracks in this round.
2. Use only the 300 original training tracks for model selection. Freeze a
   three-fold artist-grouped split with seed 20260828; record exact track IDs.
   Fit the scaler and classifier only on each fold's training portion.
3. Try the same six classifier settings for each representation:
   StandardScaler + one-vs-rest logistic regression, C in {0.01, 0.1, 1.0},
   class_weight in {None, balanced}, max_iter=2000, random_state=2026.
   Select the largest mean fold Macro AP; exact ties prefer earlier grid order.
4. For MFCC diagnosis only, hold C=1 / balanced fixed and use 25%, 50%, 100%
   of each fold's training artists (nested prefixes of a seeded permutation).
   Report actual track/artist counts and training/held-out Macro AP. These
   small curves indicate trends, not an estimate of the data needed for 80%.
5. Select thresholds using the chosen setting's out-of-fold training scores.
   Report F1-oriented thresholds using the original fixed 0.1–0.9 grid.
   Also report a precision-oriented experimental policy on a fixed grid
   0.05–0.95 in steps of 0.025. For each tag require precision >=0.80,
   recall >=0.30 and >=10 predictions in the 300 out-of-fold tracks; among
   feasible choices maximize recall, then precision, then proximity to 0.5.
   If infeasible, mark the tag unsupported and do not emit that tag under
   this policy. Abstention is NOT success and is NOT a negative diagnosis.
6. Refit each selected head on the 300 training tracks. Apply the thresholds
   without retuning to the existing 90 validation tracks. These tracks have
   been used in earlier experiments: this is development evidence, not a new
   blind evaluation. Selection and threshold estimates on the training OOF
   data are also optimistic after selecting among six configurations.
7. Report per-tag TP/FP/FN, precision, recall, F1, AP, predicted counts,
   micro/macro aggregates, and fraction of tracks with any emitted tag.
   A provisional development goal is: micro precision >=0.80, every tag's
   precision >=0.80 and recall >=0.30, >=5 predictions/tag on validation,
   and track output coverage >=0.50. These are proposed safeguards, not a
   definition of usefulness accepted by the user or a statistical guarantee.
   If any check fails, explicitly report that the goal was not met.

## Frozen candidate representation

Resample the prepared mono audio from 22050 to the encoder's required 16000
Hz. Divide the first 30 seconds into six non-overlapping 5-second chunks.
Average hidden representations over time, over the 12 transformer layers
(exclude the embedding layer), and equally over chunks: 768 features per
track. Do not search layers, crop locations or pooling rules based on scores.
Pilot inference on a learning-only preview before extracting 390 development
tracks. Keep encoder inference in evaluation mode; cache features with audio,
source and model hashes. Use a separate environment if dependencies conflict.

## Stop and release rules

After this six-setting comparison, stop and report the measured trade-offs.
Do not promote a model solely because a single precision number increases.
The old `predict.py` remains unchanged pending an explicit promotion decision.
A final performance claim requires newly preselected tracks/artists not used
for development, frozen decisions, an overlap audit, and uncertainty reporting.
Do not reuse the old 90 test tracks as a new blind test, and do not state that
80% generalizes to arbitrary user-uploaded music. No GitHub upload in this round.

References: [grouped CV](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data),
[threshold selection](https://scikit-learn.org/stable/modules/classification_threshold.html),
[MERT model card](https://huggingface.co/m-a-p/MERT-v0-public),
[MERT implementation](https://github.com/yizhilll/MERT).
