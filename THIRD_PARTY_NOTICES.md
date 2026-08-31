# Third-party data and model notices

This repository is an academic, non-commercial music-information-retrieval
portfolio. Project code, dataset metadata, audio recordings, pretrained model
weights, and the locally trained classifier heads are separate materials with
separate terms.

## MTG-Jamendo Dataset

- Source: [Music Technology Group, Universitat Pompeu Fabra](https://github.com/MTG/mtg-jamendo-dataset)
- Dataset repository code: Apache License 2.0.
- Metadata: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- Recordings: individual Creative Commons licenses recorded per track by the
  source dataset.
- Intended scope: non-commercial research and academic use.

No training or evaluation recording is included in the public repository.
Derived manifests retain source identifiers, provenance, and the metadata
license. One 30-second learning excerpt, `track_0207501_30s.wav`, is included
under its individual CC BY 3.0 license so a new user can run a deterministic
demo. Its artist, source link, processing changes, and attribution are recorded
in `data/ATTRIBUTION.md`. The other two learning recordings remain excluded.

## Discogs-MAEST

- Exact frozen repository: [`mtg-upf/discogs-maest-30s-pw-129e-519l`](https://huggingface.co/mtg-upf/discogs-maest-30s-pw-129e-519l)
- Frozen revision: `6c35f32a350f74351870937d5ae0bae1d898d1df`.
- Developer: Music Technology Group, Universitat Pompeu Fabra.

The exact 519-label repository's model card does not declare a weight license.
A related official MAEST model card declares CC BY-NC-SA 4.0 for
non-commercial use, but this project does not assume that statement cures the
missing declaration on the exact artifact. The 348 MB MAEST weights and copied
upstream files are therefore excluded from Git. The preparation script retrieves
the pinned files from the official repository and verifies their digests; each
user remains responsible for the upstream terms.

The final classifier artifact contains four sets of StandardScaler statistics
and logistic-regression parameters trained for this project. It does not
contain the MAEST encoder or any source audio. The safe NumPy artifact and its
metadata are separately licensed under CC BY-NC 4.0; see `MODEL_LICENSE.md`.

## MERT development comparisons

- [`m-a-p/MERT-v0-public`](https://huggingface.co/m-a-p/MERT-v0-public):
  CC BY-NC 4.0 according to the official model card.
- [`m-a-p/MERT-v1-95M`](https://huggingface.co/m-a-p/MERT-v1-95M):
  CC BY-NC 4.0 according to the official model card.

These weights were used only for documented development comparisons and are
excluded from Git. They are not required by the final MAEST prediction command.

## Python dependencies

Python dependencies remain under their respective upstream licenses. Installing
them does not relicense this project, its metadata, model artifacts, or audio.
Pinned runtime versions are recorded in the requirements lock files.

## Publication boundary

- Project-owned source code is covered by the repository's MIT License.
- The packaged final linear heads are covered by `MODEL_LICENSE.md`.
- Training/evaluation audio and pretrained encoder weights remain excluded
  from Git; the one attributed CC BY 3.0 demo excerpt is the stated exception.
- Dataset manifests retain provenance and source-license information.
- Upstream citations, frozen revisions, digests, and known limitations remain
  part of the release documentation.
