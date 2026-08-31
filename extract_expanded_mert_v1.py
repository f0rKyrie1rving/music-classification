"""Pilot and resumable MERT-v1 extraction for expanded development data."""

import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from mert_features import sha256
from mert_v1_features import DEST, LAYERS, WIDTH, V1MertEncoder
from prepare_dataset import save_json
from prepare_expansion import AUDIT, MANIFEST, ROOT, load_frozen


OUT = ROOT / "outputs/mert_v1"
PLAN = ROOT / "experiments/mert_v1_plan_v2.json"
FEATURES = OUT / "mert_layers.npy"
FEATURE_META = OUT / "mert_layers.json"
PROGRESS = OUT / "mert_layers_progress.json"


def development_rows():
    manifest = load_frozen()
    plan = json.loads(PLAN.read_text())
    by_id = {row["track_id"]: row for row in manifest["tracks"]}
    ids = plan["training_ids"] + plan["validation_ids"]
    if set(ids) & set(plan["excluded_test_ids"]):
        raise ValueError("Test ID entered the MERT-v1 feature plan.")
    rows = [by_id[track_id] for track_id in ids]
    if len(rows) != 1206 or any(row["split"] == "test" for row in rows):
        raise ValueError("MERT-v1 encoder rows crossed the test boundary.")
    audit = json.loads(AUDIT.read_text())
    if audit["manifest_sha256"] != sha256(MANIFEST) or set(audit["tracks"]) != set(ids):
        raise ValueError("Expanded audio audit differs from MERT-v1 rows.")
    hashes = [audit["tracks"][track_id]["wav_sha256"] for track_id in ids]
    return rows, hashes


def metadata(device):
    return {
        "encoder_manifest_sha256": sha256(DEST / "SOURCES.json"),
        "extractor_source_sha256": sha256(ROOT / "mert_v1_features.py"),
        "driver_source_sha256": sha256(Path(__file__)),
        "expanded_manifest_sha256": sha256(MANIFEST),
        "model_plan_sha256": sha256(PLAN), "device": device,
        "versions": {package: version(package) for package in
                     ("torch", "transformers", "librosa", "numpy", "soxr")},
    }


def pilot(device):
    rows, _ = development_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    encoder = V1MertEncoder(device)
    path = ROOT / rows[0]["audio_file"]
    before = time.perf_counter()
    first = encoder.extract_layers(path)
    first_seconds = time.perf_counter() - before
    before = time.perf_counter()
    second = encoder.extract_layers(path)
    second_seconds = time.perf_counter() - before
    difference = float(np.max(np.abs(first - second)))
    if first.shape != (LAYERS, WIDTH) or difference > 1e-5:
        raise ValueError("MERT-v1 pilot is invalid or non-repeatable.")
    result = {"track_id": rows[0]["track_id"], "audio_file": rows[0]["audio_file"],
              "shape": list(first.shape), "finite": bool(np.isfinite(first).all()),
              "first_seconds": first_seconds, "second_seconds": second_seconds,
              "max_abs_repeat_difference": difference, "python": platform.python_version(),
              "platform": platform.platform(), "encoder_training": encoder.model.training,
              "parameters_require_grad": any(p.requires_grad for p in encoder.model.parameters()),
              **metadata(device)}
    np.save(OUT / "mert_layers_pilot.npy", first, allow_pickle=False)
    save_json(OUT / "mert_layers_pilot.json", result)
    print(json.dumps(result, indent=2))


def extract(device):
    rows, hashes = development_rows()
    meta = metadata(device)
    pilot_result = json.loads((OUT / "mert_layers_pilot.json").read_text())
    for key, value in meta.items():
        if pilot_result[key] != value:
            raise ValueError("MERT-v1 pilot differs from this runtime and plan.")
    ids = [row["track_id"] for row in rows]
    if FEATURE_META.exists():
        final = json.loads(FEATURE_META.read_text())
        values = np.load(FEATURES, mmap_mode="r", allow_pickle=False)
        if (final["feature_sha256"] != sha256(FEATURES) or final["ids"] != ids or
                final["audio_hashes"] != hashes or values.shape != (len(rows), LAYERS, WIDTH) or
                not np.isfinite(values).all()):
            raise ValueError("Completed MERT-v1 feature cache is stale or invalid.")
        print("MERT-v1 feature cache already complete and verified.")
        return
    execution = OUT / "mert_layers_execution.json"
    if execution.exists() and json.loads(execution.read_text()) != meta:
        raise ValueError("MERT-v1 extraction was frozen with different settings.")
    save_json(execution, meta)
    completed = 0
    if PROGRESS.exists():
        progress = json.loads(PROGRESS.read_text())
        if progress["metadata"] != meta or progress["ids"] != ids or progress["audio_hashes"] != hashes:
            raise ValueError("Partial MERT-v1 feature cache is stale.")
        completed = progress["completed"]
        values = np.lib.format.open_memmap(FEATURES, mode="r+", dtype="float32",
                                           shape=(len(rows), LAYERS, WIDTH))
        if not np.isfinite(values[:completed]).all():
            raise ValueError("Completed MERT-v1 features are invalid.")
    else:
        values = np.lib.format.open_memmap(FEATURES, mode="w+", dtype="float32",
                                           shape=(len(rows), LAYERS, WIDTH))
        values[:] = np.nan
        values.flush()
        save_json(PROGRESS, {"metadata": meta, "ids": ids, "audio_hashes": hashes, "completed": 0})
    encoder = V1MertEncoder(device)
    begin, resumed = time.perf_counter(), completed
    for index, row in enumerate(rows[completed:], completed):
        values[index] = encoder.extract_layers(ROOT / row["audio_file"])
        completed = index + 1
        if completed % 10 == 0 or completed == len(rows):
            values.flush()
            save_json(PROGRESS, {"metadata": meta, "ids": ids, "audio_hashes": hashes,
                                 "completed": completed})
            print(f"MERT-v1 tracks {completed}/{len(rows)}; {time.perf_counter()-begin:.1f}s this run", flush=True)
    del values
    development_rows()
    values = np.load(FEATURES, mmap_mode="r", allow_pickle=False)
    if values.shape != (len(rows), LAYERS, WIDTH) or not np.isfinite(values).all():
        raise ValueError("Final MERT-v1 features are invalid.")
    final = {**meta, "shape": list(values.shape), "dtype": str(values.dtype), "ids": ids,
             "audio_hashes": hashes, "resumed_tracks": resumed, "new_tracks": len(rows)-resumed,
             "extraction_seconds": time.perf_counter()-begin, "test_tracks": 0,
             "feature_sha256": sha256(FEATURES)}
    save_json(FEATURE_META, final)
    print(f"Verified MERT-v1 feature cache: {final['feature_sha256']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pilot", "extract"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    (pilot if args.command == "pilot" else extract)(args.device)
