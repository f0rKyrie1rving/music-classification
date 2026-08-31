# Release artifacts

`final_heads.npy` stores only finite floating-point parameters for four
standardized logistic-regression heads. It contains no executable Python
objects, pretrained MAEST weights, training audio, or source metadata.
`final_heads.json` records the label order, thresholds, encoder revision,
evaluation-plan digest, and the SHA-256 digest of the array.

The parameters are loaded with `numpy.load(..., allow_pickle=False)`. They were
exported from the frozen final evaluation bundle by
`export_release_model.py`; the export is checked against all 239 archived
holdout feature vectors before release.

These two files are governed by [MODEL_LICENSE.md](../MODEL_LICENSE.md), not
the repository's MIT software license.
