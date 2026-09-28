"""Candidate packaging and application arithmetic, without training or audio downloads."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import candidate_predict as candidate
import predict_app
import predict_maest


WIDTH = candidate.FEATURE_WIDTH


def synthetic_arrays():
    return {"means": np.zeros((4, WIDTH)), "scales": np.ones((4, WIDTH)),
            "coefficients": np.zeros((4, WIDTH)), "intercepts": np.zeros(4),
            "offsets": np.zeros(4)}


def metadata():
    return {
        "format_version": 1, "candidate_id": "music-candidate-20260927-v1",
        "status": "candidate_not_fresh_validated", "labels": list(candidate.LABELS),
        "representation": candidate.REPRESENTATION, "feature_width": WIDTH,
        "encoder": {"model_id": candidate.MODEL_ID, "revision": candidate.REVISION,
                    "hidden_state_index": candidate.HIDDEN_STATE_INDEX},
        "thresholds": {"f1": [0.39999999999999997, .225, .2, .27499999999999997],
                       "precision_target": [.625, 1.01, 1.01, .525]},
        "precision_supported": [True, False, False, True],
        "head_provenance": [{"label": label, "kind": "synthetic"}
                            for label in candidate.LABELS],
        "baseline_reference": {"artifacts/final_heads.npy": "a" * 64},
        "license": "CC-BY-NC-4.0",
    }


class CandidatePredictionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.record = metadata()
        self.arrays = synthetic_arrays()
        for prefix, filename in candidate.REFERENCES.items():
            path = self.directory / filename
            if prefix == "weights":
                np.savez(path, **self.arrays)
            else:
                path.write_text(json.dumps({"synthetic_receipt": prefix}))
            self.record[f"{prefix}_file"] = filename
            self.record[f"{prefix}_sha256"] = candidate.sha256(path)
        self.save_metadata()

    def save_metadata(self):
        (self.directory / "candidate.json").write_text(json.dumps(self.record))

    def save_arrays(self, arrays):
        path = self.directory / "candidate.npz"
        np.savez(path, **arrays)
        self.record["weights_sha256"] = candidate.sha256(path)
        self.save_metadata()

    def bundle(self):
        return candidate.load_candidate(self.directory)

    def test_portable_bundle_needs_no_training_data_and_preserves_arrays(self):
        loaded = self.bundle()
        self.assertEqual(loaded["status"], "candidate_not_fresh_validated")
        for key, expected in self.arrays.items():
            np.testing.assert_array_equal(loaded[key], expected)
        values = candidate.predict_arrays(loaded, np.zeros((2, WIDTH)))
        self.assertEqual(values["probability"].shape, (2, 4))
        np.testing.assert_array_equal(values["raw"], values["probability"])

    def test_every_referenced_checksum_is_checked(self):
        for prefix, filename in candidate.REFERENCES.items():
            path = self.directory / filename
            original = path.read_bytes()
            with self.subTest(prefix=prefix):
                path.write_bytes(original + b"changed")
                with self.assertRaises(ValueError):
                    self.bundle()
                path.write_bytes(original)

    def test_rejects_paths_outside_bundle_and_wrong_metadata(self):
        changes = [
            ("weights_file", "../candidate.npz"), ("training_file", "/tmp/training.json"),
            ("format_version", True), ("feature_width", WIDTH + 1),
            ("labels", list(reversed(candidate.LABELS))), ("status", "released"),
            ("encoder", {}), ("precision_supported", [True] * 4),
            ("baseline_reference", {"path": "bad"}),
            ("head_provenance", [{"label": "wrong", "kind": "copied"}] * 4),
        ]
        original = copy.deepcopy(self.record)
        for key, value in changes:
            with self.subTest(key=key):
                self.record = copy.deepcopy(original)
                self.record[key] = value
                self.save_metadata()
                with self.assertRaises(ValueError):
                    self.bundle()

    def test_rejects_symlink_to_external_receipt(self):
        with tempfile.TemporaryDirectory() as outside:
            path = self.directory / "training.json"
            external = Path(outside) / "training.json"
            external.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(external)
            with self.assertRaises(ValueError):
                self.bundle()

    def test_rejects_invalid_parameter_shapes_dtype_values_or_keys(self):
        replacements = [
            ("means", np.zeros((3, WIDTH))), ("scales", np.zeros((4, WIDTH))),
            ("scales", -np.ones((4, WIDTH))),
            ("coefficients", np.zeros((4, WIDTH), dtype=np.float32)),
            ("intercepts", np.array([0., np.nan, 0., 0.])),
            ("coefficients", np.full((4, WIDTH), np.inf)),
            ("offsets", np.array([0., 0., -1., 0.])),
            ("offsets", np.zeros((1, 4))),
        ]
        for key, value in replacements:
            with self.subTest(key=key, shape=value.shape):
                arrays = synthetic_arrays()
                arrays[key] = value
                self.save_arrays(arrays)
                with self.assertRaises(ValueError):
                    self.bundle()
        arrays = synthetic_arrays()
        arrays["extra"] = np.zeros(1)
        self.save_arrays(arrays)
        with self.assertRaises(ValueError):
            self.bundle()

    def test_rejects_invalid_thresholds_and_changed_copied_policies(self):
        original = copy.deepcopy(self.record)
        changes = [
            ("f1", [0.4, .2, .2, .27499999999999997]),
            ("f1", [.39999999999999997, np.nan, .2, .27499999999999997]),
            ("f1", [.39999999999999997, 1.01, .2, .27499999999999997]),
            ("f1", [.2, .3, .4]), ("precision_target", [.625, .5, 1.01, .525]),
        ]
        for policy, thresholds in changes:
            with self.subTest(policy=policy, thresholds=thresholds):
                self.record = copy.deepcopy(original)
                self.record["thresholds"][policy] = thresholds
                self.save_metadata()
                with self.assertRaises(ValueError):
                    self.bundle()

    def test_boundary_decision_uses_unrounded_probability(self):
        bundle = self.bundle()
        threshold = bundle["thresholds"]["f1"][1]
        selected = []
        for probability in [threshold - 1e-8, threshold + 1e-8]:
            bundle["intercepts"][1] = np.log(probability) - np.log1p(-probability)
            result = candidate.predict_vector(bundle, np.zeros(WIDTH))[1]
            self.assertEqual(result["score"], result["threshold"])
            selected.append(result["selected"])
        self.assertEqual(selected, [False, True])

    def test_rejects_bad_features_policies_and_nonfinite_logits(self):
        bundle = self.bundle()
        for vector in [np.zeros(WIDTH - 1), np.zeros((1, WIDTH)),
                       np.full(WIDTH, np.nan), np.full(WIDTH, np.inf)]:
            with self.subTest(shape=vector.shape), self.assertRaises(ValueError):
                candidate.predict_vector(bundle, vector)
        for matrix in [np.zeros(WIDTH), np.zeros((2, WIDTH - 1)),
                       np.full((2, WIDTH), np.nan)]:
            with self.subTest(shape=matrix.shape), self.assertRaises(ValueError):
                candidate.predict_arrays(bundle, matrix)
        with self.assertRaises(ValueError):
            candidate.predict_vector(bundle, np.zeros(WIDTH), "unknown")
        bundle["coefficients"][:] = 1e300
        with self.assertRaises(ValueError):
            candidate.predict_vector(bundle, np.full(WIDTH, 1e300))

    def test_extreme_finite_logits_and_selective_suppression(self):
        bundle = self.bundle()
        for value in [-1e300, 1e300]:
            bundle["intercepts"][:] = value
            arrays = candidate.predict_arrays(bundle, np.zeros((2, WIDTH)), "precision_target")
            self.assertTrue(np.isfinite(arrays["probability"]).all())
            self.assertFalse(arrays["decision"][:, 1:3].any())
            if value > 0:
                np.testing.assert_array_equal(arrays["decision"][0], [True, False, False, True])

    def test_row_arithmetic_matches_real_installed_heads_bit_for_bit(self):
        old = predict_maest.load_bundle()
        bundle = self.bundle()
        for name in ["means", "scales", "coefficients", "intercepts"]:
            bundle[name] = np.asarray(old[name], dtype=np.float64).copy()
        bundle["thresholds"] = copy.deepcopy(old["thresholds"])
        features = np.random.default_rng(73).normal(size=(9, WIDTH))
        actual = candidate.predict_arrays(bundle, features)
        for index, feature in enumerate(features):
            expected = np.sum(((feature - old["means"]) / old["scales"])
                              * old["coefficients"], axis=1) + old["intercepts"]
            np.testing.assert_array_equal(actual["logits"][index], expected)
            np.testing.assert_array_equal(actual["raw"][index],
                                          np.exp(-np.logaddexp(0., -expected)))
            old_details = predict_app.predict_vector(old, feature, score_mode="raw")
            self.assertEqual(candidate.predict_vector(bundle, feature), old_details)

    def test_baseline_comparison_extracts_once_and_shares_identical_feature(self):
        feature = np.random.default_rng(9).normal(size=WIDTH)

        class Encoder:
            calls = 0

            def extract(self, audio):
                self.calls += 1
                return feature

        encoder = Encoder()
        with patch.object(predict_app, "predict_vector", wraps=predict_app.predict_vector) as baseline:
            result = candidate.classify("synthetic.wav", encoder=encoder,
                                        directory=self.directory, compare_baseline=True)
        self.assertEqual(encoder.calls, 1)
        self.assertIs(baseline.call_args.args[1], feature)
        self.assertEqual(result["scores"], candidate.predict_vector(self.bundle(), feature))
        old = predict_maest.load_bundle()
        correction = predict_app.load_correction(old, predict_app.METADATA, predict_app.WEIGHTS,
                                                 predict_app.PLAN, predict_app.CORRECTION)
        self.assertEqual(result["baseline"]["scores"],
                         predict_app.predict_vector(old, feature, correction=correction))
        self.assertTrue(result["baseline"]["shared_feature"])


if __name__ == "__main__":
    unittest.main()
