"""Unit checks for the final user-audio prediction entry point."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from maest_hf_features import FEATURE_WIDTH
from predict_maest import load_bundle, predict_vector


class MaestPredictionTests(unittest.TestCase):
    def bundle(self):
        return {
            "means": np.zeros((4, FEATURE_WIDTH)),
            "scales": np.ones((4, FEATURE_WIDTH)),
            "coefficients": np.zeros((4, FEATURE_WIDTH)),
            "intercepts": np.array([0.0, -1.38629436112, -0.40546510811, -0.84729786039]),
            "thresholds": {
                "f1": [0.4, 0.275, 0.375, 0.275],
                "precision_target": [0.625, 1.01, 1.01, 0.525],
            },
        }

    def test_multilabel_threshold_decisions(self):
        result = predict_vector(self.bundle(), np.zeros(FEATURE_WIDTH))
        self.assertEqual([item["label"] for item in result if item["selected"]],
                         ["electronic", "ambient", "rock"])
        self.assertEqual(result[0]["score"], 0.5)

    def test_selective_policy_and_invalid_features(self):
        result = predict_vector(self.bundle(), np.ones(FEATURE_WIDTH), "precision_target")
        self.assertFalse(any(item["selected"] for item in result))
        for feature in (np.zeros(FEATURE_WIDTH - 1), np.full(FEATURE_WIDTH, np.nan)):
            with self.subTest(shape=feature.shape), self.assertRaises(ValueError):
                predict_vector(self.bundle(), feature)
        with self.assertRaises(ValueError):
            predict_vector(self.bundle(), np.zeros(FEATURE_WIDTH), "unknown")

    def test_packaged_artifact_is_safe_and_validated(self):
        bundle = load_bundle()
        self.assertEqual(bundle["means"].shape, (4, FEATURE_WIDTH))
        self.assertEqual(bundle["labels"], ["electronic", "pop", "ambient", "rock"])
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.npy"
            changed.write_bytes(Path("artifacts/final_heads.npy").read_bytes() + b"x")
            with self.assertRaises(ValueError):
                load_bundle(weights_path=changed)


if __name__ == "__main__":
    unittest.main()
