# Official MTG-UPF Hugging Face MAEST experiment

Frozen after the MERT-v1 development outcome and the failed UPF direct-download
attempt, but before MAEST weights, embeddings, downstream scores, or
development-validation outcomes were inspected.

## Candidate

Use `mtg-upf/discogs-maest-30s-pw-129e-519l` from the official Music Technology
Group organization on Hugging Face, pinned to revision
`6c35f32a350f74351870937d5ae0bae1d898d1df`. The MAEST authors' official
repository identifies Hugging Face as an inference source. The repository uses
Transformers' AST implementation, an Apache-2.0 feature-extractor source file,
and a 347,827,260-byte safetensors checkpoint with upstream SHA-256:

`881cec6abdcb6ef986367e6c0db02dc760cc0f88a52b625cb9d6e2b54d505548`

Official sources:

- https://huggingface.co/mtg-upf/discogs-maest-30s-pw-129e-519l
- https://github.com/palonso/MAEST
- https://essentia.upf.edu/models.html#maest

The Hugging Face model card does not declare a model-weight license. Treat the
weights conservatively under Essentia's stated CC BY-NC-ND 4.0 model terms: use
them only for this non-commercial experiment, keep them out of Git, and retain
attribution. The reviewed feature-extractor file itself declares Apache 2.0.

## Fixed preprocessing and representation

- Read exactly the first 30 source seconds with `soundfile`, average channels,
  and resample to exactly 480,000 samples at 16 kHz using pinned `soxr` HQ.
  Reject short, silent, corrupt, or non-finite inputs.
- Load the reviewed official feature extractor locally, with no remote-code or
  network access during inference. Verify its fixed 96-bin mel settings,
  1,876-frame maximum, normalization mean/std, 512-point FFT, and hop 256.
- Load `ASTForAudioClassification` from verified safetensors only, freeze every
  parameter, and run in evaluation/inference mode on CPU.
- Request all hidden states and take `hidden_states[7]`, which is the output of
  zero-based transformer block 6: the seventh transformer block specified by
  the MAEST authors.
- Concatenate its CLS token, DIST token, and the mean of all remaining signal
  tokens. This is one fixed 2,304-wide vector per track.
- Do not search layers, use direct 519-label outputs, combine MERT features,
  tune preprocessing, or add augmentation.

## Unchanged development boundary

- Reuse the exact 903 training tracks, 303 development-validation tracks,
  labels, artist-grouped folds, and 329 excluded test IDs from
  `experiments/expanded_model_plan.json`.
- Do not download, encode, inspect, fit, or score test audio.
- Search the existing six logistic-regression settings only. Select per label
  by mean AP over the same five grouped training folds.
- Derive the same F1 and precision-target thresholds from training OOF scores
  only, then apply each policy once to the 303 development-validation tracks.

A pre-outcome pilot must run the first frozen development track twice and verify
finite, deterministic 2,304-wide output, evaluation mode, and no trainable
parameters. Any further implementation correction before downstream scores
requires a new note and plan hash.

## Decision rule

The provisional usefulness goal remains micro precision >= 0.80, every label
precision >= 0.80 and recall >= 0.30, at least 10 validation predictions per
label, and output coverage >= 0.50. Report all metrics and do not promote a
candidate because a single number rises.

If this candidate fails, keep the holdout sealed and reassess target-label
quality, ontology, and the attainable precision/coverage tradeoff before trying
another representation.

## Limitation

The pretraining data are not public at track level, so overlap with MTG-Jamendo
cannot be audited. Any eventual result must disclose that it is not a proven
pretraining-disjoint evaluation.
