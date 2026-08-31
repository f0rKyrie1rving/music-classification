"""Verify every selected excerpt and freeze its local acquisition evidence."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from prepare_dataset import CHUNK, leading_id3_size, save_json
from train import validate_manifest

ROOT = Path(__file__).resolve().parent


def main():
    manifest_path = ROOT / "data/dataset_manifest.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    validate_manifest(manifest)
    rows = manifest["tracks"]
    expected = {"train": 300, "validation": 90, "test": 90}
    if Counter(r["split"] for r in rows) != expected:
        raise ValueError("The frozen 300/90/90 selection is incomplete or changed.")
    received, skipped, full_verified, rms_values = 0, 0, 0, []
    for row in rows:
        if "nd" in row["license_code"].lower():
            raise ValueError("Unexpected NoDerivatives recording in this subset.")
        receipt_path = ROOT / "data/source/download_receipts" / f'{row["track_id"]}.json'
        receipt = json.loads(receipt_path.read_text())
        if receipt["track_id"] != row["track_id"]:
            raise ValueError("Receipt belongs to a different track.")
        segments = receipt.get("byte_ranges") or [dict(
            byte_range=receipt["byte_range"], sha256=receipt["prefix_sha256"],
            cache_file=f'data/source/audio_prefixes/{row["track_id"]}.mp3.part')]
        archive, count, previous_end = row["archive"], 0, -1
        for segment in sorted(segments, key=lambda s: s["byte_range"][0]):
            start, end = segment["byte_range"]
            if not archive["offset"] <= start <= end < archive["offset"] + archive["size"] or start <= previous_end:
                raise ValueError("Downloaded range is out of bounds or overlaps another range.")
            content = (ROOT / segment["cache_file"]).read_bytes()
            if len(content) != end - start + 1 or hashlib.sha256(content).hexdigest() != segment["sha256"]:
                raise ValueError("Downloaded segment length/hash mismatch.")
            count += len(content)
            previous_end = end
        if count != receipt["downloaded_bytes"] or count > 20 * CHUNK:
            raise ValueError("Acquisition exceeds the declared byte budget.")
        if len(segments) > 1:
            header = (ROOT / segments[0]["cache_file"]).read_bytes()[:10]
            if segments[1]["byte_range"][0] != archive["offset"] + leading_id3_size(header):
                raise ValueError("Audio segment does not begin after the documented ID3 tag.")
            if receipt["full_mp3_sha256_verified"]:
                raise ValueError("A noncontiguous excerpt cannot verify the complete MP3.")
            skipped += 1
        if receipt["full_mp3_sha256_verified"]:
            if count != archive["size"] or segments[0]["sha256"] != row["expected_full_sha256"]:
                raise ValueError("Incorrect full-file verification claim.")
            full_verified += 1
        audio_path = ROOT / row["audio_file"]
        info = sf.info(audio_path)
        if (info.frames, info.samplerate, info.channels, info.subtype) != (661500, 22050, 1, "PCM_16"):
            raise ValueError("Unexpected WAV format.")
        y, _ = sf.read(audio_path, dtype="float64")
        rms = float(np.sqrt(np.mean(y**2)))
        if not np.isfinite(y).all() or rms < 1e-5:
            raise ValueError("Invalid or near-silent excerpt.")
        rms_values.append(rms)
        received += count
        row["wav_sha256"] = receipt["wav_sha256"]
        row["acquisition"] = dict(receipt, byte_ranges=segments)
    for split, cap, minimum in [("train", 3, 50), ("validation", 2, 15), ("test", 2, 15)]:
        subset = [r for r in rows if r["split"] == split]
        if max(Counter(r["artist_id"] for r in subset).values()) > cap:
            raise ValueError("Artist cap exceeded.")
        if min(sum(r["targets"][j] for r in subset) for j in range(len(manifest["labels"]))) < minimum:
            raise ValueError("Label coverage is below the frozen minimum.")
        if sum(not any(r["targets"]) for r in subset) != round(len(subset) * 0.1):
            raise ValueError("Background quota differs from the plan.")
    planned_path = ROOT / "data/source/planned_manifest.json"
    if not planned_path.exists():
        planned_path.write_bytes(original)
    save_json(manifest_path, manifest)
    audit = dict(verified_tracks=len(rows), split_counts=expected,
                 artist_count=len({r["artist_id"] for r in rows}),
                 cached_source_bytes=received, audio_seconds=len(rows) * 30,
                 id3_tags_skipped=skipped, full_mp3_hashes_verified=full_verified,
                 min_rms=min(rms_values), max_rms=max(rms_values),
                 manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                 verified_at_utc=datetime.now(timezone.utc).isoformat())
    save_json(ROOT / "data/dataset_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
