"""Unit tests for deterministic audio feature extraction."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from features import (FEATURE_NAMES, MEL_SETTINGS, N_MFCC, SAMPLE_RATE,
                      analyze_audio, extract_features, summarize_mfcc)
from mert_features import load_chunks
from mert_layer_features import LAYERS, summarize_layer_chunks
from mert_v1_features import load_v1_chunks, summarize_v1_chunks
from maest_features import FEATURE_WIDTH, summarize_maest_embeddings, validate_audio
from maest_hf_features import load_audio as load_maest_audio, summarize_hidden_state


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.tone = 0.2 * np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE) / SAMPLE_RATE)

    def write(self, values, name="test.wav", rate=SAMPLE_RATE):
        path = self.folder / name
        sf.write(path, values, rate, subtype="FLOAT")
        return path

    def test_shape_order_and_temporal_statistics(self):
        mfcc = np.arange(N_MFCC)[:, None] + np.array([0.0, 2.0, 4.0])
        vector = summarize_mfcc(mfcc)
        np.testing.assert_allclose(vector[:N_MFCC], np.arange(N_MFCC) + 2)
        np.testing.assert_allclose(vector[N_MFCC:], np.sqrt(8 / 3))
        self.assertEqual(len(set(FEATURE_NAMES)), 26)
        self.assertEqual(FEATURE_NAMES[0], "mfcc_00_mean")
        self.assertEqual(FEATURE_NAMES[-1], "mfcc_12_std")

    def test_maest_audio_and_token_summary(self):
        audio = np.ones(480001, dtype=np.float32) * 0.1
        self.assertEqual(validate_audio(audio).shape, (480000,))
        tokens = np.arange(4 * 768, dtype=float).reshape(1, 1, 4, 768)
        vector = summarize_maest_embeddings(tokens)
        self.assertEqual(vector.shape, (FEATURE_WIDTH,))
        np.testing.assert_array_equal(vector[:768], tokens[0, 0, 0])
        np.testing.assert_array_equal(vector[768:1536], tokens[0, 0, 1])
        np.testing.assert_array_equal(vector[1536:], tokens[0, 0, 2:].mean(axis=0))
        with self.assertRaises(ValueError):
            validate_audio(np.zeros(480000, dtype=np.float32))
        with self.assertRaises(ValueError):
            summarize_maest_embeddings(np.zeros((2, 767)))
        np.testing.assert_array_equal(vector, summarize_hidden_state(tokens[0]))

    def test_maest_user_audio_first_thirty_seconds_and_validation(self):
        rate = 16000
        tone = 0.2 * np.sin(2 * np.pi * 440 * np.arange(30 * rate) / rate)
        exact = self.write(tone, "maest_30s.wav", rate)
        longer = self.write(np.concatenate((tone, np.ones(rate))), "maest_31s.wav", rate)
        np.testing.assert_array_equal(load_maest_audio(exact), load_maest_audio(longer))
        self.assertEqual(load_maest_audio(exact).shape, (480000,))
        for values in (tone[:-1], np.zeros(30 * rate)):
            with self.subTest(size=len(values)), self.assertRaises(ValueError):
                load_maest_audio(self.write(values, "maest_invalid.wav", rate))
        with self.assertRaises(FileNotFoundError):
            load_maest_audio(self.folder / "missing_maest.wav")

    def test_tone_shapes_finite_and_repeatable(self):
        path = self.write(self.tone)
        mel, mfcc = analyze_audio(path)
        frames = 1 + (SAMPLE_RATE - MEL_SETTINGS["n_fft"]) // MEL_SETTINGS["hop_length"]
        self.assertEqual(mel.shape, (64, frames))
        self.assertEqual(mfcc.shape, (13, frames))
        vector = extract_features(path)
        self.assertEqual(vector.shape, (26,))
        self.assertTrue(np.isfinite(vector).all())
        self.assertTrue((vector[13:] >= 0).all())
        np.testing.assert_array_equal(vector, extract_features(path))
        np.testing.assert_array_equal(vector, summarize_mfcc(mfcc))

    def test_stereo_and_resampling(self):
        path = self.write(np.column_stack((self.tone, self.tone)))
        mono = self.write(self.tone, "mono.wav")
        np.testing.assert_allclose(extract_features(path), extract_features(mono))
        rate = 44100
        tone = 0.2 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)
        vector = extract_features(self.write(tone, "44100.wav", rate))
        self.assertEqual(vector.shape, (26,))
        self.assertTrue(np.isfinite(vector).all())

    def test_only_first_thirty_seconds_are_used(self):
        first = np.tile(self.tone, 30)
        short = self.write(first, "30s.wav")
        long = self.write(np.concatenate((first, np.ones(SAMPLE_RATE))), "31s.wav")
        np.testing.assert_array_equal(extract_features(short), extract_features(long))

    def test_reject_empty_short_silent_and_nonfinite(self):
        cases = [np.array([]), self.tone[:100], np.zeros(SAMPLE_RATE),
                 np.full(SAMPLE_RATE, 1e-7), np.full(SAMPLE_RATE, np.nan)]
        for values in cases:
            with self.subTest(size=values.size):
                with self.assertRaises(ValueError):
                    extract_features(self.write(values))

    def test_reject_missing_and_corrupt_file(self):
        with self.assertRaises(FileNotFoundError):
            extract_features(self.folder / "missing.wav")
        path = self.folder / "corrupt.wav"
        path.write_bytes(b"This is not audio.")
        with self.assertRaises((RuntimeError, ValueError)):
            extract_features(path)

    def test_mert_chunks_preserve_time_order_and_ignore_tail(self):
        first = np.repeat(np.arange(1, 7, dtype=np.float32) / 10, 80000)
        path = self.write(np.concatenate((first, np.ones(16000))), rate=16000)
        chunks = load_chunks(path)
        self.assertEqual(chunks.shape, (6, 80000))
        np.testing.assert_array_equal(chunks, first.reshape(6, 80000))

    def test_mert_stereo_resamples_to_same_mono_chunks(self):
        tone = np.tile(self.tone, 30)
        mono = self.write(tone, "mert_mono.wav")
        stereo = self.write(np.column_stack((tone, tone)), "mert_stereo.wav")
        np.testing.assert_array_equal(load_chunks(mono), load_chunks(stereo))
        self.assertEqual(load_chunks(mono).shape, (6, 80000))

    def test_mert_rejects_short_silent_and_nonfinite_inputs(self):
        for values in (self.tone, np.zeros(30 * 16000), np.full(30 * 16000, np.nan)):
            with self.subTest(size=values.size), self.assertRaises(ValueError):
                load_chunks(self.write(values, rate=16000))

    def test_layer_chunk_summary_preserves_layers(self):
        values = np.zeros((6, LAYERS, 768), dtype=np.float32)
        for chunk in range(6):
            values[chunk] = chunk + np.arange(LAYERS)[:, None]
        summary = summarize_layer_chunks(values)
        self.assertEqual(summary.shape, (LAYERS, 768))
        np.testing.assert_array_equal(summary[:, 0], np.arange(LAYERS) + 2.5)
        with self.assertRaises(ValueError):
            summarize_layer_chunks(values[:, :-1])

    def test_mert_v1_chunks_use_24khz_and_ignore_tail(self):
        first = np.repeat(np.arange(1, 7, dtype=np.float32) / 10, 120000)
        path = self.write(np.concatenate((first, np.ones(24000))), rate=24000)
        chunks = load_v1_chunks(path)
        self.assertEqual(chunks.shape, (6, 120000))
        np.testing.assert_array_equal(chunks, first.reshape(6, 120000))

    def test_mert_v1_chunks_resample_exact_30_seconds(self):
        values = np.tile(self.tone, 30)
        chunks = load_v1_chunks(self.write(values, "v1_22050.wav", rate=22050))
        self.assertEqual(chunks.shape, (6, 120000))
        self.assertTrue(np.isfinite(chunks).all())

    def test_mert_v1_layer_summary(self):
        values = np.zeros((6, 13, 768), dtype=np.float32)
        for chunk in range(6):
            values[chunk] = chunk + np.arange(13)[:, None]
        summary = summarize_v1_chunks(values)
        np.testing.assert_array_equal(summary[:, 0], np.arange(13) + 2.5)
        with self.assertRaises(ValueError):
            summarize_v1_chunks(values[:, :-1])


if __name__ == "__main__":
    unittest.main()
