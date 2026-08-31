# Pinned MERT source review

Reviewed 2026-08-28, before importing the downloaded model source.

Source: [m-a-p/MERT-v0-public](https://huggingface.co/m-a-p/MERT-v0-public),
revision `e8413e398b3180ca488534aea68fd829402044a4`.

Downloaded all six needed files from the author's host over verified HTTPS.
The weight file is 377,552,987 bytes; its SHA-256 matches the author's LFS
metadata: `9b25bde740483579d9895f35d074a949f6593ef48449b6d76e26ee3c0e5e9acb`.
Small files match the official Git blob IDs. Exact checks are embedded in
`prepare_mert.py`; local receipts are in `models/mert-v0-public/SOURCES.json`.

## Source inspection

Read the complete `configuration_MERT.py` and `modeling_MERT.py`, along with
the model/preprocessor JSON and model card. The source defines a modified
HuBERT encoder with optional CQT/deep-normalization branches. No shell,
network, credential-reading, dynamic-eval or file-mutation code was found
in these two files. This inspection is not a blanket security guarantee for
third-party dependencies.

The pinned config uses 12 layers, width 768, 16,000 Hz, float32, with CQT and
deepnorm disabled. Therefore the optional `nnAudio` import is not required
for this checkpoint, despite the upstream import-time warning. The feature
extractor explicitly sets `do_normalize=false`; preserve this setting.

Load only the checked local Python files. Read the checkpoint with
`torch.load(..., map_location="cpu", weights_only=True)` and require a strict
state-dictionary match. Never fall back to unrestricted unpickling or to
unreviewed moving remote code. Inference runs in evaluation/inference mode
and does not upload audio. A later CPU pilot on this Mac succeeded with an
identical repeated 768-dimensional vector; the completed development run is
documented in [MERT_RESULTS.md](../MERT_RESULTS.md). No GPU claim is made.

## License and evaluation limits

The model card identifies CC BY-NC 4.0 for this release. Its identity and
attribution remain distinct from project code, MTG-Jamendo data, and the final
classifier artifact, as recorded in `THIRD_PARTY_NOTICES.md`.

The model card reports pretraining on Music4All and part of FMA. The project
has not audited track/artist/audio overlap with MTG-Jamendo. A good result on the
development subset must not be represented as an overlap-free blind test.

The representation, split, six classifier settings and threshold policy were
already fixed in [IMPROVEMENT_PROTOCOL.md](IMPROVEMENT_PROTOCOL.md). No layer,
chunk location or pooling choice may be changed after observing results in
this round.
