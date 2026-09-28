"""Boundary tests for immutable retrospective MERT feature caches."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research.calibration_extension.cache import (
    complete_cache, immutable_json, mean_transformer_layers, sha256,
    unique_ids, validate_arrays,
)


class CacheBoundaryTests(unittest.TestCase):
    def test_float32_pooling_excludes_embedding_layer(self):
        rng = np.random.default_rng(204)
        layers = rng.normal(size=(13, 768)).astype(np.float64)
        expected = layers.astype(np.float32)[1:].mean(axis=0)
        actual = mean_transformer_layers(layers)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.float32)
        layers[0] = 10000
        np.testing.assert_array_equal(mean_transformer_layers(layers), expected)

    def test_invalid_layer_shape_or_nonfinite_rejected(self):
        for layers in (np.ones((12, 768)), np.full((13, 768), np.nan)):
            with self.assertRaises(ValueError):
                mean_transformer_layers(layers)

    def test_duplicate_and_misordered_ids_rejected(self):
        with self.assertRaises(ValueError):
            unique_ids(["a", "a"])
        with self.assertRaises(ValueError):
            validate_arrays(np.zeros((2, 768), np.float32), ["a", "b"], ["b", "a"])

    def test_partial_progress_bounds_and_finite_prefix(self):
        x = np.full((2, 768), np.nan, np.float32)
        x[0] = 1
        validate_arrays(x, ["a", "b"], ["a", "b"], 1)
        for completed in (-1, 3, 2, 1.5):
            with self.assertRaises(ValueError):
                validate_arrays(x, ["a", "b"], ["a", "b"], completed)

    def test_float64_cache_rejected(self):
        with self.assertRaises(ValueError):
            validate_arrays(np.zeros((1, 768)), ["a"], ["a"])

    def test_immutable_receipt_rejects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            immutable_json(path, {"a": 1})
            immutable_json(path, {"a": 1})
            with self.assertRaises(ValueError):
                immutable_json(path, {"a": 2})

    def test_completed_cache_digest_and_freeze_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            receipt = Path(directory) / "receipt.json"
            self.assertFalse(complete_cache(path, receipt, ["a"], "frozen"))
            np.savez(path, x=np.zeros((1, 768), np.float32), ids=np.array(["a"]))
            with self.assertRaises(ValueError):
                complete_cache(path, receipt, ["a"], "frozen")
            receipt.write_text(json.dumps({"freeze_sha256": "frozen", "feature_sha256": sha256(path)}))
            self.assertTrue(complete_cache(path, receipt, ["a"], "frozen"))
            with self.assertRaises(ValueError):
                complete_cache(path, receipt, ["a"], "different")
            with path.open("ab") as stream:
                stream.write(b"mutation")
            with self.assertRaises(ValueError):
                complete_cache(path, receipt, ["a"], "frozen")


if __name__ == "__main__":
    unittest.main()
