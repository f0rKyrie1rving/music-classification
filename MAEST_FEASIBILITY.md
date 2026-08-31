# MAEST local feasibility check

Checked: 2026-08-30. This check installed an isolated runtime and did not load a
MAEST model, extract dataset features, fit a classifier, or make any test
prediction.

## Why this candidate

MERT-v1 improved the four-label development result only modestly. MAEST is a
more targeted next candidate because it was trained explicitly to predict music
styles rather than only serving as a general music encoder. The official
Essentia documentation recommends `discogs-maest-30s-pw` for downstream tasks
and describes the newer `discogs-maest-30s-pw-519l` as trained on 4 million full
tracks and 519 Discogs styles, with slightly better performance expected.

Official sources:

- https://essentia.upf.edu/models.html#maest
- https://essentia.upf.edu/models/feature-extractors/maest/
- https://essentia.upf.edu/licensing_information.html
- https://pypi.org/project/essentia-tensorflow/2.1b6.dev1389/

## Runtime result

The machine is macOS 15.7.9 on ARM64 with CPython 3.13.15. The current latest
Essentia-TensorFlow release targets CPython 3.14, so this check pinned the older
official wheel that matches CPython 3.13 and macOS 15 ARM64:

`essentia_tensorflow-2.1b6.dev1389-cp313-cp313-macosx_15_0_arm64.whl`

- Downloaded size: 115,255,077 bytes.
- PyPI SHA-256 and locally observed SHA-256:
  `2a172f9eaf49a034e508ffc76d64c18f34e3a88944d721e8b00f7c1f56d89bd5`.
- Isolated environment: `.venv-maest`.
- Fixed dependencies: NumPy 2.5.2, PyYAML 6.0.3, six 1.17.0.
- Import result: `MonoLoader`, `TensorflowPredictMAEST`, and
  `TensorflowPredict2D` are all available.

The local compatibility gate therefore passed. Model inference still requires
a separate single-track pilot after downloading the pinned official weights.

## Distribution constraint

The Essentia-TensorFlow package declares `AGPL-3.0-only`. Essentia's official
licensing page says its pretrained models are CC BY-NC-ND 4.0 for
non-commercial use. This is compatible with a non-commercial admissions
experiment, but it changes how the GitHub project should be packaged:

- do not commit or redistribute the model weights;
- download them from the official UPF URL and record their checksum;
- retain attribution and non-commercial notices;
- if the final application imports Essentia, review the repository's code
  license for AGPL compatibility before publishing.

This is a technical license audit, not legal advice. MAEST remains an
experimental candidate until it passes the frozen development protocol.

## Selected implementation after network check

The UPF direct-download host could not be reached from this machine through the
command line, in-app browser, or controlled Chrome. Before any MAEST model or
embedding was obtained, the experiment switched to the MAEST authors' official
`mtg-upf` Hugging Face repository. That distribution provides the same 30-second
519-label architecture as safetensors plus an Apache-2.0 feature extractor, so
the isolated Essentia environment above is retained only as an audited
feasibility check and is not the selected v2 runtime. See
`experiments/MAEST_HF_IMPLEMENTATION_NOTE.md` and
`experiments/MAEST_HF_PROTOCOL.md`.
