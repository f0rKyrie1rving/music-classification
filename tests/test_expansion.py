"""Safeguards for the expanded-data role boundary."""

import unittest
from collections import Counter

import numpy as np

from expanded_model import ONTOLOGY, broad_targets, representation
from final_maest_holdout import bootstrap_intervals, select_blind_queries
from prepare_expansion import phase_role
from repair_short_audio import normalized_excerpt


class ExpansionRoleTests(unittest.TestCase):
    def test_development_roles(self):
        self.assertEqual(phase_role("train", False), "train")
        self.assertEqual(phase_role("validation", False), "development_validation")

    def test_test_rows_are_never_development(self):
        self.assertEqual(phase_role("test", True), "historical_test_excluded")
        self.assertEqual(phase_role("test", False), "new_holdout")

    def test_unknown_split_is_rejected(self):
        with self.assertRaises(ValueError):
            phase_role("other", False)

    def test_broad_ontology_is_multilabel_and_conservative(self):
        row = {"tags": ["poprock", "techno"]}
        self.assertEqual(broad_targets(row), [1, 1, 0, 1])
        self.assertEqual(broad_targets({"tags": ["alternative", "dance"]}), [0, 0, 0, 0])
        self.assertEqual(set(ONTOLOGY), {"electronic", "pop", "ambient", "rock"})

    def test_layer_representation_selection(self):
        values = np.arange(2 * 13 * 3).reshape(2, 13, 3)
        np.testing.assert_array_equal(representation(values, "layer_12"), values[:, 12])
        np.testing.assert_array_equal(representation(values, "mean_transformer"), values[:, 1:].mean(axis=1))
        with self.assertRaises(ValueError):
            representation(values, "layer_13")

    def test_short_source_repair_is_strictly_bounded(self):
        repaired, source_gap, output_gap = normalized_excerpt(np.ones((2999, 1)), 100)
        self.assertEqual(source_gap, 1)
        self.assertEqual(len(repaired), 661500)
        self.assertGreaterEqual(output_gap, 0)
        with self.assertRaises(ValueError):
            normalized_excerpt(np.ones((2989, 1)), 100)

    def test_final_blind_queries_are_balanced_unique_and_repeatable(self):
        rows = []
        labels = list(ONTOLOGY)
        for index in range(80):
            label = labels[(index // 10) % 4] if index < 40 else None
            rows.append({"track_id": f"track_{index:03d}", "artist": f"artist {index}",
                         "title": f"title {index}", "audio_file": f"audio/{index}.wav",
                         "tags": [label] if label else []})
        first = select_blind_queries(rows, seed=17)
        second = select_blind_queries(rows, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 40)
        self.assertEqual(len({row["track_id"] for row in first}), 40)
        self.assertEqual(Counter((row["label"], row["source_target"]) for row in first),
                         Counter({(label, target): 5 for label in labels for target in (0, 1)}))

    def test_final_bootstrap_intervals_are_bounded(self):
        targets = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0],
                            [0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
        scores = targets * 0.7 + 0.15
        result = bootstrap_intervals(targets, scores, np.full(4, 0.5), 20, 11)
        self.assertEqual(result["replicates"], 20)
        for interval in result["overall"].values():
            self.assertLessEqual(0.0, interval[0])
            self.assertLessEqual(interval[0], interval[1])
            self.assertLessEqual(interval[1], 1.0)


if __name__ == "__main__":
    unittest.main()
