"""Freeze, acquire, and verify the expanded development data without test use.

The original 480-track manifest remains untouched.  The new holdout IDs are
frozen here, but their audio is deliberately not downloaded by the default
development command.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from prepare_dataset import download_track, save_json


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/source"
POOL = SOURCE / "training_pool.json"
ORIGINAL = ROOT / "data/dataset_manifest.json"
PROTOCOL = ROOT / "experiments/DATA_EXPANSION_PROTOCOL.md"
MANIFEST = ROOT / "data/expanded_manifest.json"
AUDIT = ROOT / "data/expanded_development_audit.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase_role(split, historical_test):
    if split == "train":
        return "train"
    if split == "validation":
        return "development_validation"
    if split == "test":
        return "historical_test_excluded" if historical_test else "new_holdout"
    raise ValueError(f"Unexpected official split: {split}")


def load_indexes(pool):
    indexed, hashes = {}, {}
    for shard in pool["candidate_archives"]:
        path = SOURCE / f"archive_index_{shard}.json"
        index = json.loads(path.read_text())
        if not index["complete"]:
            raise ValueError(f"Archive index {shard} is incomplete.")
        hashes[shard] = sha256(path)
        for name, entry in index["members"].items():
            if name in indexed:
                raise ValueError(f"Duplicate archive member: {name}")
            indexed[name] = dict(**entry, archive_url=index["archive_url"],
                                 archive_bytes=index["archive_bytes"])
    return indexed, hashes


def freeze():
    if MANIFEST.exists():
        raise FileExistsError("Expanded manifest already frozen; do not resample it.")
    pool = json.loads(POOL.read_text())
    original = json.loads(ORIGINAL.read_text())
    indexed, index_hashes = load_indexes(pool)
    rows = pool["records"]
    if len(rows) != len({row["track_id"] for row in rows}):
        raise ValueError("Duplicate track IDs in source pool.")
    if set(indexed) != {row["archive_member"] for row in rows}:
        raise ValueError("Complete indexes do not exactly cover the frozen source pool.")
    original_by_id = {row["track_id"]: row for row in original["tracks"]}
    historical_test = {row["track_id"] for row in original["tracks"] if row["split"] == "test"}
    selected = []
    for source_row in rows:
        row = dict(source_row)
        row["archive"] = indexed[row["archive_member"]]
        row["targets"] = [int(label in row["tags"]) for label in pool["labels"]]
        row["phase_role"] = phase_role(row["split"], row["track_id"] in historical_test)
        previous = original_by_id.get(row["track_id"])
        if previous:
            for key in ("artist_id", "archive_member", "split", "tags", "targets"):
                if previous[key] != row[key]:
                    raise ValueError(f"Original row changed for {row['track_id']}: {key}")
            row["audio_file"] = previous["audio_file"]
            row["existing_wav_sha256"] = previous["wav_sha256"]
        else:
            row["audio_file"] = f"data/expanded_audio/{row['track_id']}.wav"
        selected.append(row)
    expected = {"train": 903, "development_validation": 303,
                "historical_test_excluded": 90, "new_holdout": 239}
    if Counter(row["phase_role"] for row in selected) != expected:
        raise ValueError("Expansion role counts differ from the frozen protocol.")
    artists = {role: {row["artist_id"] for row in selected if row["phase_role"] == role}
               for role in expected}
    if artists["train"] & (artists["development_validation"] | artists["historical_test_excluded"] |
                            artists["new_holdout"]):
        raise ValueError("Official split artist leakage into training.")
    manifest = {
        "dataset": pool["dataset"],
        "official_split": "split-0",
        "labels": pool["labels"],
        "roles": expected,
        "protocol_sha256": sha256(PROTOCOL),
        "source_pool_sha256": sha256(POOL),
        "original_manifest_sha256": sha256(ORIGINAL),
        "preparation_source_sha256": sha256(Path(__file__)),
        "archive_index_sha256": index_hashes,
        "audio_policy": "Non-ND Creative Commons audio for non-commercial research; individual licenses retained.",
        "excerpt": {"offset_seconds": 0, "duration_seconds": 30, "sample_rate": 22050,
                    "channels": 1, "subtype": "PCM_16"},
        "tracks": selected,
    }
    save_json(MANIFEST, manifest)
    print(f"Frozen expanded manifest with {len(selected)} tracks: {expected}")


def load_frozen():
    manifest = json.loads(MANIFEST.read_text())
    checks = ((PROTOCOL, "protocol_sha256"), (POOL, "source_pool_sha256"),
              (ORIGINAL, "original_manifest_sha256"), (Path(__file__), "preparation_source_sha256"))
    for path, key in checks:
        if sha256(path) != manifest[key]:
            raise ValueError(f"Frozen input changed: {path.relative_to(ROOT)}")
    for shard, expected in manifest["archive_index_sha256"].items():
        if sha256(SOURCE / f"archive_index_{shard}.json") != expected:
            raise ValueError(f"Frozen archive index changed: {shard}")
    return manifest


def acquire_development(workers):
    manifest = load_frozen()
    rows = [row for row in manifest["tracks"]
            if row["phase_role"] in {"train", "development_validation"}
            and "existing_wav_sha256" not in row]
    failures, completed = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(download_track, row): row for row in rows}
        for future in as_completed(jobs):
            row = jobs[future]
            try:
                completed.append(future.result())
            except Exception as error:
                failures.append({"track_id": row["track_id"], "error": str(error)})
                print(f"FAILED {row['track_id']}: {error}", flush=True)
            if (len(completed) + len(failures)) % 25 == 0:
                print(f"Development audio {len(completed)}/{len(rows)}; failures={len(failures)}", flush=True)
    save_json(SOURCE / "expanded_download_status.json", {
        "requested": len(rows), "completed": len(completed), "failures": failures,
        "holdout_downloaded": 0,
    })
    if failures:
        raise SystemExit("Incomplete download; rerun after inspecting expanded_download_status.json.")
    print(f"Acquired {len(completed)} new development excerpts; holdout audio remains untouched.")


def verify_development():
    manifest = load_frozen()
    rows = [row for row in manifest["tracks"]
            if row["phase_role"] in {"train", "development_validation"}]
    records, rms_values = {}, []
    for row in rows:
        path = ROOT / row["audio_file"]
        digest = sha256(path)
        expected = row.get("existing_wav_sha256")
        if expected is None:
            receipt = json.loads((SOURCE / "download_receipts" / f"{row['track_id']}.json").read_text())
            expected = receipt["wav_sha256"]
        if digest != expected:
            raise ValueError(f"WAV checksum mismatch: {row['track_id']}")
        info = sf.info(path)
        if (info.frames, info.samplerate, info.channels, info.subtype) != (661500, 22050, 1, "PCM_16"):
            raise ValueError(f"Unexpected WAV format: {row['track_id']}")
        y, _ = sf.read(path, dtype="float32")
        rms = float(np.sqrt(np.mean(y.astype("float64") ** 2)))
        if not np.isfinite(y).all() or rms < 1e-5:
            raise ValueError(f"Invalid or near-silent audio: {row['track_id']}")
        rms_values.append(rms)
        records[row["track_id"]] = {"wav_sha256": digest, "rms": rms}
    result = {
        "manifest_sha256": sha256(MANIFEST),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "roles": dict(Counter(row["phase_role"] for row in rows)),
        "verified_tracks": len(rows),
        "min_rms": min(rms_values),
        "max_rms": max(rms_values),
        "new_holdout_audio_verified": 0,
        "tracks": records,
    }
    save_json(AUDIT, result)
    print(json.dumps({key: result[key] for key in result if key != "tracks"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "download-development", "verify-development"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "download-development":
        acquire_development(args.workers)
    else:
        verify_development()


if __name__ == "__main__":
    main()
