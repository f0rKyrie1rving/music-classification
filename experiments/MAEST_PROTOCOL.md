# Discogs-MAEST representation experiment

Frozen after the MERT-v1 development outcome was known and before MAEST model
weights, embeddings, downstream scores, or development-validation outcomes were
inspected.

## Candidate and rationale

Use the official UPF/MTG `discogs-maest-30s-pw-519l-2.pb` model, released
2025-01-22. Its official metadata describes a TensorFlow 2.17 model trained on
4 million full tracks with a multi-label objective over 519 Discogs styles.
The 30-second PaSST-initialized model is chosen because the official Essentia
documentation recommends the 30-second `pw` family and says the updated 519
label model is expected to perform slightly better.

Pinned official files:

- model: https://essentia.upf.edu/models/feature-extractors/maest/discogs-maest-30s-pw-519l-2.pb
- metadata: https://essentia.upf.edu/models/feature-extractors/maest/discogs-maest-30s-pw-519l-2.json
- expected model size from the official index: 347,996,675 bytes.

The official model license is CC BY-NC-ND 4.0 for non-commercial use. The model
will stay local and will not be redistributed. Inference uses the isolated
Essentia-TensorFlow 2.1b6.dev1389 environment described in
`MAEST_FEASIBILITY.md`; that dependency is AGPL-3.0-only.

## One fixed representation

- Decode only the first 30 seconds as mono at 16 kHz with Essentia's official
  `MonoLoader` settings (`resampleQuality=4`). Reject empty, silent, non-finite,
  and shorter inputs rather than silently changing them.
- Request `PartitionedCall/Identity_7`, the seventh transformer-layer embedding
  named as the embedding output in the model metadata and recommended in the
  official example.
- Following the official downstream recommendation, concatenate the `CLS`
  token, the `DIST` token, and the mean of all remaining signal tokens. The
  expected fixed vector width is 3 x 768 = 2,304.
- Do not search other MAEST layers, combine MERT features, use the 519 direct
  predictions, tune preprocessing, or add augmentation in this experiment.

Using embeddings rather than mapping the 519 Discogs outputs avoids changing
the project's four broad-label definition mid-experiment.

## Unchanged development boundary

- Reuse the exact 903 training tracks, 303 development-validation tracks,
  labels, artist-grouped folds, and 329 excluded test IDs frozen in
  `experiments/expanded_model_plan.json`.
- Do not download, encode, inspect, fit, or score test audio.
- Search only the existing six logistic-regression settings for each label.
  Select by mean average precision over the same five grouped training folds.
- Choose the same F1-oriented and precision-target thresholds from training
  out-of-fold scores only. Apply them once to the 303 development-validation
  tracks without retuning.

Before full extraction, a pilot must verify one frozen development track twice:
the output is finite, exactly 2,304 wide, deterministic, and obtained without
training or weight mutation. Any implementation correction before downstream
scores must be recorded in a separate note and a new plan hash.

## Decision rule

The provisional usefulness goal is unchanged: micro precision >= 0.80, every
label precision >= 0.80 and recall >= 0.30, at least 10 validation predictions
per label, and output coverage >= 0.50. Report all metrics even if this goal
fails. Do not promote a candidate based on one improving number.

If MAEST also fails, keep the holdout sealed. Reassess label quality, ontology,
and the attainable operating point rather than trying another encoder without a
new evidence-based hypothesis.

## Known limitation

The official metadata identifies the pretraining source as an unreleased
Discogs23 dataset. Its overlap with the Jamendo tracks cannot be audited from
public metadata. This must be disclosed when interpreting development and any
future holdout result; the experiment cannot claim a proven pretraining-disjoint
external evaluation.
