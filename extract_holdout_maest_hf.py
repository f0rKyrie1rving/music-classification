"""Resumably extract the fixed MAEST representation for final holdout audio."""

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from maest_hf_features import DEST, FEATURE_WIDTH, HfMaestEncoder
from mert_features import sha256
from prepare_dataset import save_json
from prepare_expansion import MANIFEST, ROOT
from prepare_holdout import AUDIT, holdout_rows


PLAN = ROOT / "experiments/final_holdout_plan.json"
OUT = ROOT / "outputs/final_maest"
FEATURES = OUT / "holdout_features.npy"
META = OUT / "holdout_features.json"
PROGRESS = OUT / "holdout_features_progress.json"


def rows_and_hashes():
    plan, rows = holdout_rows(); audit = json.loads(AUDIT.read_text())
    if (audit["plan_sha256"] != sha256(PLAN) or audit["manifest_sha256"] != sha256(MANIFEST)
            or audit["verified_tracks"] != len(rows) or audit["historical_test_audio_used"] != 0
            or set(audit["tracks"]) != set(plan["holdout_ids"])):
        raise ValueError("Final holdout audio audit is missing or invalid.")
    return rows, [audit["tracks"][row["track_id"]]["wav_sha256"] for row in rows]


def metadata(device):
    return {"plan_sha256": sha256(PLAN), "holdout_audit_sha256": sha256(AUDIT),
            "expanded_manifest_sha256": sha256(MANIFEST),
            "extractor_source_sha256": sha256(Path(__file__)),
            "feature_source_sha256": sha256(ROOT / "maest_hf_features.py"),
            "encoder_manifest_sha256": sha256(DEST / "SOURCES.json"), "device": device,
            "versions": {name: version(name) for name in
                         ("torch", "transformers", "safetensors", "numpy", "soundfile", "soxr")}}


def extract(device):
    rows, hashes = rows_and_hashes(); meta = metadata(device); OUT.mkdir(parents=True, exist_ok=True)
    ids = [row["track_id"] for row in rows]
    if META.exists():
        final = json.loads(META.read_text()); values = np.load(FEATURES, mmap_mode="r", allow_pickle=False)
        if (final["feature_sha256"] != sha256(FEATURES) or final["ids"] != ids
                or final["audio_hashes"] != hashes or values.shape != (239, FEATURE_WIDTH)
                or not np.isfinite(values).all() or final["historical_test_tracks"] != 0):
            raise ValueError("Completed final holdout feature cache is invalid.")
        print("Final holdout MAEST feature cache already complete and verified."); return
    execution = OUT / "holdout_execution.json"
    if execution.exists() and json.loads(execution.read_text()) != meta:
        raise ValueError("Holdout extraction runtime differs from its first invocation.")
    save_json(execution, meta); completed = 0
    if PROGRESS.exists():
        progress = json.loads(PROGRESS.read_text())
        if progress["metadata"] != meta or progress["ids"] != ids or progress["audio_hashes"] != hashes:
            raise ValueError("Partial holdout feature cache is stale.")
        completed = progress["completed"]
        values = np.lib.format.open_memmap(
            FEATURES, mode="r+", dtype="float32", shape=(len(rows), FEATURE_WIDTH))
        if not np.isfinite(values[:completed]).all():
            raise ValueError("Completed holdout feature rows are invalid.")
    else:
        values = np.lib.format.open_memmap(
            FEATURES, mode="w+", dtype="float32", shape=(len(rows), FEATURE_WIDTH))
        values[:] = np.nan; values.flush()
        save_json(PROGRESS, {"metadata": meta, "ids": ids, "audio_hashes": hashes, "completed": 0})
    encoder = HfMaestEncoder(device); begin, resumed = time.perf_counter(), completed
    for index, row in enumerate(rows[completed:], completed):
        values[index] = encoder.extract(ROOT / row["audio_file"]); completed = index + 1
        if completed % 10 == 0 or completed == len(rows):
            values.flush(); save_json(PROGRESS, {"metadata": meta, "ids": ids,
                "audio_hashes": hashes, "completed": completed})
            print(f"Holdout MAEST tracks {completed}/{len(rows)}; "
                  f"{time.perf_counter()-begin:.1f}s this run", flush=True)
    del values; rows_and_hashes(); values = np.load(FEATURES, mmap_mode="r", allow_pickle=False)
    if values.shape != (239, FEATURE_WIDTH) or not np.isfinite(values).all():
        raise ValueError("Final holdout feature matrix is invalid.")
    final = {**meta, "shape": list(values.shape), "dtype": str(values.dtype), "ids": ids,
             "audio_hashes": hashes, "resumed_tracks": resumed, "new_tracks": len(rows)-resumed,
             "extraction_seconds": time.perf_counter()-begin, "historical_test_tracks": 0,
             "feature_sha256": sha256(FEATURES)}
    save_json(META, final); print(f"Verified final MAEST features: {final['feature_sha256']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args(); extract(args.device)
