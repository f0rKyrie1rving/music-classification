"""Create feature-inspection outputs for the local learning samples."""

import csv
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib
import numpy as np
from features import (FEATURE_NAMES, MAX_SECONDS, MEL_SETTINGS, MIN_SECONDS,
                      N_MFCC, SAMPLE_RATE, SILENCE_RMS, analyze_audio,
                      summarize_mfcc)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa.display


def plot_spectrogram(log_mel, sample, output):
    """The display has a per-excerpt reference; MFCCs use the fixed reference."""
    times = librosa.frames_to_time(np.arange(log_mel.shape[1]), sr=SAMPLE_RATE,
                                  hop_length=MEL_SETTINGS["hop_length"],
                                  n_fft=MEL_SETTINGS["n_fft"])
    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    picture = librosa.display.specshow(
        log_mel - log_mel.max(), x_coords=times, sr=SAMPLE_RATE,
        hop_length=MEL_SETTINGS["hop_length"], x_axis="time", y_axis="mel",
        fmin=0, fmax=SAMPLE_RATE / 2, vmin=-80, vmax=0, cmap="magma", ax=ax)
    ax.set(title=f'{sample["title"]} — {sample["artist"]}\nMel spectrogram · 64 frequency bands',
           xlabel="Time in excerpt (seconds)", ylabel="Frequency (Hz, Mel scale)",
           xlim=(0, sample["preview"]["duration_seconds"]))
    fig.colorbar(picture, ax=ax, label="Power relative to excerpt peak (dB)")
    fig.text(0.5, -0.03,
             "Source: MTG-Jamendo · Original audio 30–60 s · CC BY 3.0 · See data/ATTRIBUTION.md",
             ha="center", fontsize=8, color="#4b5563")
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    samples = json.loads((ROOT / "data/sample_manifest.json").read_text())["samples"]
    rows, example = [], None
    print("Extracting audio features; the first run may take a moment.", flush=True)
    for sample in samples:
        log_mel, mfcc = analyze_audio(ROOT / sample["preview"]["file"])
        values = summarize_mfcc(mfcc)
        rows.append(dict(track_id=sample["track_id"], title=sample["title"],
                         **dict(zip(FEATURE_NAMES, values, strict=True))))
        if sample["track_id"] == "track_0913702":
            example = (log_mel, sample)
        print(f'{sample["title"]}: {mfcc.shape[1]} analysis windows -> {values.size} features', flush=True)
    if not rows or example is None:
        raise ValueError("The learning manifest must include the spectrogram example.")

    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    with (output / "sample_features.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["track_id", "title", *FEATURE_NAMES])
        writer.writeheader()
        writer.writerows(rows)
    config = dict(sample_rate=SAMPLE_RATE, mono=True, offset_seconds=0,
                  max_seconds=MAX_SECONDS, min_seconds=MIN_SECONDS, silence_rms=SILENCE_RMS,
                  mel_settings=MEL_SETTINGS, n_mfcc=N_MFCC, dct_type=2, norm="ortho",
                  lifter=0, power_reference=1.0, top_db=80.0, std_ddof=0,
                  res_type="soxr_hq", dtype="float64", feature_names=FEATURE_NAMES,
                  python=platform.python_version(), purpose="Learning samples only",
                  versions={name: version(name) for name in
                            ("librosa", "numpy", "scipy", "soundfile", "soxr", "matplotlib")})
    (output / "feature_config.json").write_text(json.dumps(config, indent=2) + "\n")
    plot_spectrogram(*example, output / "spectrogram.png")
    print(f"Feature matrix: {len(rows)} tracks x {len(FEATURE_NAMES)} features (not a training set).")
    print(f"Saved outputs to: {output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Could not inspect features: {error}") from None
