"""Print metadata for the local WAV learning samples."""

import argparse
import json
import wave
from pathlib import Path


def describe_audio(path):
    """Read basic audio properties using Python's standard library."""
    with wave.open(str(path), "rb") as audio:
        sample_rate = audio.getframerate()
        frames = audio.getnframes()
        return {
            "file": path.name,
            "duration_seconds": round(frames / sample_rate, 3),
            "sample_rate_hz": sample_rate,
            "channels": audio.getnchannels(),
            "bits_per_sample": audio.getsampwidth() * 8,
            "audio_frames": frames,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, help="WAV files to inspect")
    args = parser.parse_args()
    preview_dir = Path(__file__).resolve().parent / "data" / "previews"
    paths = args.files or sorted(preview_dir.glob("*.wav"))
    if not paths:
        parser.error("No WAV previews found. Download and prepare the samples first.")
    for path in paths:
        try:
            print(json.dumps(describe_audio(path), ensure_ascii=False))
        except (OSError, EOFError, wave.Error) as error:
            parser.exit(1, f"Could not read {path}: {error}\n")


if __name__ == "__main__":
    main()
