"""Unit tests for feature comparison and partition boundaries."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from compare_features import counts, development_rows
from extra_features import EXTENDED_NAMES, extract_extended_features
from features import SAMPLE_RATE, extract_features


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.tone = 0.2 * np.sin(2 * np.pi * 440 * np.arange(2 * SAMPLE_RATE) / SAMPLE_RATE)

    def write(self, values, name="test.wav", rate=SAMPLE_RATE):
        path = self.folder / name
        sf.write(path, values, rate, subtype="FLOAT")
        return path

    def test_shape_names_baseline_and_repeatability(self):
        path = self.write(self.tone)
        vector = extract_extended_features(path)
        self.assertEqual(vector.shape, (46,))
        self.assertEqual(len(set(EXTENDED_NAMES)), 46)
        np.testing.assert_array_equal(vector[:26], extract_features(path))
        np.testing.assert_array_equal(vector, extract_extended_features(path))
        self.assertTrue(np.isfinite(vector).all())
        self.assertTrue((vector[30:34] >= 0).all())

    def test_tone_frequency_pitch_class_and_noise_flatness(self):
        vector = extract_extended_features(self.write(self.tone))
        self.assertAlmostEqual(vector[26], 440, delta=10)
        self.assertEqual(np.argmax(vector[34:]), 9)  # A in C-through-B chroma order.
        noise = np.random.default_rng(2026).normal(0, 0.1, self.tone.size)
        noisy = extract_extended_features(self.write(noise, "noise.wav"))
        self.assertGreater(noisy[28], vector[28])

    def test_stereo_resampling_and_first_thirty_seconds(self):
        mono = extract_extended_features(self.write(self.tone))
        stereo = extract_extended_features(self.write(np.column_stack((self.tone, self.tone)), "stereo.wav"))
        np.testing.assert_array_equal(mono, stereo)
        high_rate = 0.2 * np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)
        vector = extract_extended_features(self.write(high_rate, "high.wav", 44100))
        self.assertAlmostEqual(vector[26], 440, delta=10)
        first = np.tile(self.tone, 15)
        short = extract_extended_features(self.write(first, "30s.wav"))
        long = extract_extended_features(self.write(np.r_[first, np.ones(SAMPLE_RATE)], "31s.wav"))
        np.testing.assert_array_equal(short, long)

    def test_invalid_inputs_rejected(self):
        for values in [np.zeros(SAMPLE_RATE), self.tone[:100], np.full(SAMPLE_RATE, np.nan)]:
            with self.subTest(size=values.size), self.assertRaises(ValueError):
                extract_extended_features(self.write(values))
        with self.assertRaises(FileNotFoundError):
            extract_extended_features(self.folder / "missing.wav")

    def test_test_rows_excluded_and_artist_overlap_rejected(self):
        rows = [dict(track_id=str(i), artist_id=str(i), split=s)
                for i, s in enumerate(["train", "validation", "test"])]
        self.assertEqual(development_rows(dict(tracks=rows)), rows[:2])
        rows[2]["artist_id"] = rows[0]["artist_id"]
        with self.assertRaisesRegex(ValueError, "Artist leakage"):
            development_rows(dict(tracks=rows))

    def test_false_positive_false_negative_and_threshold_boundary(self):
        y = np.array([[1], [0], [1], [0]])
        score = np.array([[.5], [.9], [.1], [.2]])
        self.assertEqual(counts(y, score, np.array([.5])), [dict(tp=1, fp=1, fn=1, tn=1)])


if __name__ == "__main__":
    unittest.main()
