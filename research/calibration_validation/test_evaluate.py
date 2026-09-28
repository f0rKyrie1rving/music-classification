"""Synthetic invariants for saved-model inference; no new validation results."""
import copy
import inspect
import unittest

import numpy as np
from scipy.special import expit

from research.calibration.common import LABELS
from .evaluate import (FEATURE_DIM, logits_from_parameters,
                       probabilities_from_parameters, validate_feature_order,
                       validate_observation_partition)


def fixture():
    heads = {}
    fits = {}
    policy = {"labels": {}}
    for j, label in enumerate(LABELS):
        mean = np.zeros(FEATURE_DIM)
        scale = np.ones(FEATURE_DIM)
        coef = np.zeros((1, FEATURE_DIM))
        mean[:2], scale[:2], coef[0, :2] = [2., -3.], [2., 4.], [j + 1., -2.]
        heads[label] = {"scaler_mean": mean.tolist(), "scaler_scale": scale.tolist(),
                        "coef": coef.tolist(), "intercept": [.5 * j], "classes": [0, 1]}
        fits[label] = {"slope": .6 + .2 * j, "intercept": -.7 + .1 * j, "success": True}
        policy["labels"][label] = {"selected_lambda": [0., .25, .5, 1.][j], "full_fit": copy.deepcopy(fits[label])}
    return {"heads": heads, "calibrators": {"sigmoid": fits}}, policy


class SavedInferenceTests(unittest.TestCase):
    def test_reconstructs_scaling_and_signed_linear_head(self):
        parameters, _ = fixture()
        x = np.zeros((3, FEATURE_DIM), dtype=np.float32)
        x[:, :2] = [[4., 1.], [2., -3.], [-2., 5.]]
        expected = np.column_stack([(x[:, 0] - 2) / 2 * (j + 1) - (x[:, 1] + 3) / 2 + .5 * j for j in range(4)])
        np.testing.assert_array_equal(logits_from_parameters(x, parameters), expected)

    def test_invalid_classifier_state_is_rejected(self):
        parameters, _ = fixture()
        x = np.zeros((1, FEATURE_DIM), dtype=np.float32)
        for field, value in (("scaler_scale", [0.] * FEATURE_DIM), ("classes", [1, 0]), ("coef", [[0.]])):
            bad = copy.deepcopy(parameters)
            bad["heads"][LABELS[0]][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                logits_from_parameters(x, bad)
        with self.assertRaises(ValueError):
            logits_from_parameters(np.full_like(x, np.nan), parameters)

    def test_blends_are_exact_and_preserve_label_ranking(self):
        parameters, policy = fixture()
        z = np.tile(np.array([-12., -2., 0., 1., 8.])[:, None], (1, 4))
        result = probabilities_from_parameters(z, parameters, policy)
        np.testing.assert_array_equal(result["identity"], expit(z))
        np.testing.assert_array_equal(result["conservative_sigmoid"][:, 0], result["identity"][:, 0])
        np.testing.assert_array_equal(result["conservative_sigmoid"][:, 3], result["sigmoid"][:, 3])
        for j, strength in enumerate([0., .25, .5, 1.]):
            np.testing.assert_allclose(result["conservative_sigmoid"][:, j],
                (1 - strength) * result["identity"][:, j] + strength * result["sigmoid"][:, j], rtol=0, atol=0)
        for p in result.values():
            self.assertTrue(np.all(np.diff(p, axis=0) > 0))
            self.assertTrue(np.all((p >= 0) & (p <= 1)))
        self.assertFalse(np.allclose(result["conservative_sigmoid"].sum(axis=1), 1))

    def test_policy_parameter_drift_and_invalid_strength_are_rejected(self):
        parameters, policy = fixture()
        z = np.zeros((2, 4))
        for key, value in (("selected_lambda", .3), ("full_fit", {"slope": 3., "intercept": 1., "success": True})):
            bad = copy.deepcopy(policy)
            bad["labels"][LABELS[0]][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                probabilities_from_parameters(z, parameters, bad)

    def test_features_must_match_order_dtype_shape_and_finite_values(self):
        rows = [{"track_id": "x", "artist_id": "a", "tags": ["ambient"]},
                {"track_id": "y", "artist_id": "b", "tags": []}]
        x = np.zeros((2, FEATURE_DIM), dtype=np.float32)
        ids, artists = validate_feature_order(rows, x, {"ids": ["x", "y"]})
        self.assertEqual(ids.tolist(), ["x", "y"])
        self.assertEqual(artists.tolist(), ["a", "b"])
        bad_nan = x.copy(); bad_nan[0, 0] = np.nan
        cases = [(rows, x, {"ids": ["y", "x"]}), (rows, x.astype(np.float64), {"ids": ["x", "y"]}),
                 (rows, x[:1], {"ids": ["x", "y"]}), (rows, bad_nan, {"ids": ["x", "y"]}),
                 ([rows[0], rows[0]], x, {"ids": ["x", "x"]})]
        for data, features, metadata in cases:
            with self.subTest(metadata=metadata, dtype=features.dtype, shape=features.shape), self.assertRaises(ValueError):
                validate_feature_order(data, features, metadata)

    def test_inference_has_no_new_answers_or_fitting_argument(self):
        self.assertEqual(list(inspect.signature(logits_from_parameters).parameters), ["features", "parameters"])
        self.assertEqual(list(inspect.signature(probabilities_from_parameters).parameters), ["logits", "parameters", "policy"])

    def test_observation_partition_rejects_unexplained_or_changed_rows(self):
        chosen = [{"track_id": str(i), "artist_id": f"artist_{i}", "tags": []} for i in range(4)]
        observed = [chosen[0], chosen[2]]
        acquisition = {"requested": 4, "replacement_tracks": 0,
                       "successful": [{"track_id": str(i)} for i in (0, 1, 2)],
                       "failures": [{"track_id": "3", "error": "download failed"}]}
        extraction = {"state": "complete", "selected_tracks": 4, "selected_artists": 4,
                      "observed_tracks": 2, "observed_artists": 2,
                      "download_failures": copy.deepcopy(acquisition["failures"]),
                      "extraction_failures": [{"track_id": "1", "error": "decode failed"}]}
        validate_observation_partition(chosen, observed, acquisition, extraction)
        altered = copy.deepcopy(observed); altered[0]["tags"] = ["ambient"]
        for bad_rows in (observed[:1], observed[::-1], observed + [observed[0]], altered):
            with self.subTest(rows=bad_rows), self.assertRaises(ValueError):
                validate_observation_partition(chosen, bad_rows, acquisition, extraction)
        bad_extraction = copy.deepcopy(extraction)
        bad_extraction["extraction_failures"] = []
        with self.assertRaises(ValueError):
            validate_observation_partition(chosen, observed, acquisition, bad_extraction)
        bad_acquisition = copy.deepcopy(acquisition)
        bad_acquisition["successful"].append({"track_id": "0"})
        with self.assertRaises(ValueError):
            validate_observation_partition(chosen, observed, bad_acquisition, extraction)


if __name__ == "__main__":
    unittest.main()
