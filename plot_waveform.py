"""Plot the waveform of one local learning sample."""

import os
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Save an image; no GUI or display server is required.
import matplotlib.pyplot as plt


def read_audio(path):
    """Read mono PCM16 WAV samples on a fixed full-scale amplitude axis."""
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise ValueError("This exercise requires a mono, 16-bit PCM WAV.")
        sample_rate = audio.getframerate()
        frames = audio.getnframes()
        raw = audio.readframes(frames)
    if frames == 0 or len(raw) != frames * 2:
        raise ValueError("The WAV is empty or its sample data is incomplete.")
    # Fixed scaling, NOT normalization to each track's loudest sample.
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    return samples, sample_rate


def main():
    path = ROOT / "data" / "previews" / "track_0913702_30s.wav"
    samples, sample_rate = read_audio(path)
    time = np.arange(samples.size) / sample_rate
    duration = samples.size / sample_rate
    if duration < 5.01:
        raise ValueError("The example needs at least 5.01 seconds of audio.")

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.8), layout="constrained")
    fig.suptitle("Back In The Days — Chill Carrier", fontsize=16)
    axes[0].plot(time, samples, color="#146e87", linewidth=0.35)
    axes[0].set(title="Full excerpt · 22,050 samples per second",
                xlabel="Time in excerpt (seconds)", xlim=(0, duration))
    axes[0].axvline(5, color="#b64e24", linewidth=1)
    axes[0].text(5.2, 0.8, "Detail below starts here", color="#874023", fontsize=9)

    # A 10 ms window: markers show individual stored sample values.
    zoom = (time >= 5) & (time < 5.01)
    axes[1].plot((time[zoom] - 5) * 1000, samples[zoom], color="#b64e24",
                 linewidth=0.8, marker="o", markersize=2)
    axes[1].set(title="Detail · first 10 milliseconds after the 5-second mark",
                xlabel="Time since the 5-second mark (milliseconds)", xlim=(0, 10))
    for axis in axes:
        axis.set(ylabel="Sample amplitude\n(fraction of full scale)", ylim=(-1, 1))
        axis.axhline(0, color="#63717d", linewidth=0.5)
        axis.grid(alpha=0.18)
    fig.text(0.5, -0.025,
             "Source: MTG-Jamendo · Original audio 30–60 s · CC BY 3.0 · See data/ATTRIBUTION.md",
             ha="center", fontsize=8, color="#4b5563")

    output = ROOT / "outputs" / "waveform.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Read {samples.size:,} samples ({duration:g} seconds at {sample_rate:,} Hz).")
    print(f"Saved: {output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, EOFError, wave.Error, ValueError) as error:
        raise SystemExit(f"Could not plot audio: {error}") from None
