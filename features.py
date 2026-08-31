"""Shared MFCC feature extraction for the future training and prediction code."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache" / "numba"))

import librosa
import numpy as np

SAMPLE_RATE = 22050
MAX_SECONDS = 30
MIN_SECONDS = 1
SILENCE_RMS = 1e-5
N_MFCC = 13
MEL_SETTINGS = dict(n_fft=2048, hop_length=512, n_mels=64, center=False,
                    window="hann", power=2.0, fmin=0, fmax=SAMPLE_RATE / 2,
                    htk=False, norm="slaney")
FEATURE_NAMES = [f"mfcc_{i:02d}_{stat}"
                 for stat in ("mean", "std") for i in range(N_MFCC)]


def analyze_audio(path):
    """Analyze the first <=30 s of the input; return log-Mel power and MFCCs."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, offset=0,
                             duration=MAX_SECONDS, dtype=np.float64,
                             res_type="soxr_hq")
    except librosa.util.exceptions.ParameterError as error:
        raise ValueError(f"Invalid audio: {error}") from error
    if y.size < MIN_SECONDS * sr:
        raise ValueError("Audio must contain at least one second.")
    if not np.isfinite(y).all():
        raise ValueError("Audio contains non-finite sample values.")
    if np.sqrt(np.mean(y ** 2)) < SILENCE_RMS:
        raise ValueError("Audio is silent or below the RMS threshold (1e-5).")
    mel = librosa.feature.melspectrogram(y=y, sr=sr, **MEL_SETTINGS)
    log_mel = librosa.power_to_db(mel, ref=1.0, top_db=80.0)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC,
                                dct_type=2, norm="ortho", lifter=0)
    if not np.isfinite(mfcc).all():
        raise ValueError("MFCC calculation produced non-finite values.")
    return log_mel, mfcc


def summarize_mfcc(mfcc):
    """Return 13 temporal means, then 13 population standard deviations."""
    return np.concatenate((mfcc.mean(axis=1), mfcc.std(axis=1, ddof=0)))


def extract_features(path):
    """Return 26 numbers in FEATURE_NAMES order; labels are never inputs."""
    _, mfcc = analyze_audio(path)
    return summarize_mfcc(mfcc)
