"""Fixed 46-feature candidate for the validation-only comparison."""

import librosa
import numpy as np

from features import (FEATURE_NAMES, MAX_SECONDS, MEL_SETTINGS, SAMPLE_RATE,
                      analyze_audio, summarize_mfcc)

SERIES_NAMES = ["centroid_hz", "rolloff85_power_hz", "flatness", "onset_strength"]
EXTRA_NAMES = [f"{name}_{stat}" for stat in ("mean", "std") for name in SERIES_NAMES]
EXTRA_NAMES += [f"chroma_{note}_mean" for note in
                ("C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B")]
EXTENDED_NAMES = FEATURE_NAMES + EXTRA_NAMES


def extract_extended_features(path):
    # Reuse the baseline's validation, log-Mel calculation and MFCC statistics.
    log_mel, mfcc = analyze_audio(path)
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, offset=0,
                        duration=MAX_SECONDS, dtype=np.float64, res_type="soxr_hq")
    stft_args = {key: MEL_SETTINGS[key] for key in ("n_fft", "hop_length", "window", "center")}
    magnitude = np.abs(librosa.stft(y, **stft_args))
    power = magnitude ** 2
    series = np.vstack([
        librosa.feature.spectral_centroid(S=magnitude, sr=SAMPLE_RATE)[0],
        librosa.feature.spectral_rolloff(S=power, sr=SAMPLE_RATE, roll_percent=0.85)[0],
        librosa.feature.spectral_flatness(S=power, power=1.0, amin=1e-10)[0],
        librosa.onset.onset_strength(S=log_mel, sr=SAMPLE_RATE, lag=1, max_size=1,
                                    center=False, detrend=False, aggregate=np.mean),
    ])
    chroma = librosa.feature.chroma_stft(S=power, sr=SAMPLE_RATE, tuning=0.0,
                                       n_chroma=12, norm=np.inf, **stft_args)
    vector = np.concatenate((summarize_mfcc(mfcc), series.mean(axis=1),
                             series.std(axis=1, ddof=0), chroma.mean(axis=1)))
    if vector.shape != (len(EXTENDED_NAMES),) or not np.isfinite(vector).all():
        raise ValueError("Invalid extended feature vector.")
    return vector
