"""Repair only verified full source files that fall <=0.1 s short of 30 s.

This post-freeze exception is intentionally narrow.  It cannot replace tracks,
fetch more data, accept corrupt hashes, or pad a materially short recording.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from prepare_dataset import leading_id3_size, save_json
from prepare_expansion import MANIFEST, ROOT, SOURCE, load_frozen


MAX_GAP_SECONDS = 0.1


def normalized_excerpt(samples, rate):
    samples = np.asarray(samples)
    if samples.ndim != 2 or rate <= 0 or not np.isfinite(samples).all():
        raise ValueError("Invalid decoded source samples.")
    source_target = 30 * rate
    source_gap = source_target - len(samples)
    if not 0 < source_gap <= round(MAX_GAP_SECONDS * rate):
        raise ValueError("Source gap is outside the <=0.1-second repair rule.")
    mono = samples.mean(axis=1)
    y = soxr.resample(mono, rate, 22050, quality="HQ") if rate != 22050 else mono
    if len(y) > 661500:
        y = y[:661500]
    output_gap = 661500 - len(y)
    if not 0 <= output_gap <= round(MAX_GAP_SECONDS * 22050):
        raise ValueError("Resampled gap is outside the repair rule.")
    return np.pad(y, (0, output_gap)), source_gap, output_gap


def repair(row):
    track_id = row["track_id"]
    source = SOURCE / "audio_prefixes" / f"{track_id}.mp3.part"
    archive = row["archive"]
    if source.stat().st_size != archive["size"]:
        raise ValueError(f"Repair requires the complete archive member: {track_id}")
    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_hash != row["expected_full_sha256"]:
        raise ValueError(f"Complete MP3 checksum mismatch: {track_id}")
    samples, rate = sf.read(source, dtype="float64", always_2d=True)
    y, source_gap, output_gap = normalized_excerpt(samples, rate)
    dest = ROOT / row["audio_file"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(".tmp.wav")
    sf.write(temporary, y, 22050, subtype="PCM_16")
    temporary.replace(dest)
    wav_hash = hashlib.sha256(dest.read_bytes()).hexdigest()
    start = archive["offset"]
    receipt = {
        "track_id": track_id, "downloaded_bytes": len(source_bytes),
        "byte_ranges": [{"byte_range": [start, start + len(source_bytes) - 1],
                         "sha256": source_hash, "cache_file": str(source.relative_to(ROOT))}],
        "leading_id3_bytes": leading_id3_size(source_bytes[:10]), "skipped_id3_bytes": 0,
        "prefix_sha256": source_hash, "full_mp3_sha256_verified": True,
        "wav_sha256": wav_hash, "frames": 661500, "sample_rate": 22050,
        "original_sample_rate": rate, "source_padding_rule": "verified full source <=0.1 seconds short",
        "source_frames_missing": source_gap, "output_frames_padded": output_gap,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(SOURCE / "download_receipts" / f"{track_id}.json", receipt)
    return receipt


def main():
    manifest = load_frozen()
    status = json.loads((SOURCE / "expanded_download_status.json").read_text())
    failures = status["failures"]
    if not failures:
        print("No failed downloads require the short-source exception.")
        return
    if any("Not enough decodable audio within bounded prefix" not in item["error"] for item in failures):
        raise ValueError("At least one failure is not eligible for short-source repair.")
    by_id = {row["track_id"]: row for row in manifest["tracks"]}
    repaired = [repair(by_id[item["track_id"]]) for item in failures]
    save_json(SOURCE / "expanded_download_status.json", {
        "requested": status["requested"], "completed": status["completed"] + len(repaired),
        "failures": [], "holdout_downloaded": 0,
        "post_freeze_short_source_repairs": [item["track_id"] for item in repaired],
    })
    print(json.dumps(repaired, indent=2))


if __name__ == "__main__":
    main()
