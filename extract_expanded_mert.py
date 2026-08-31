"""Pilot and resumable layer-wise MERT extraction for expanded development data."""

import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from mert_features import DEST, sha256
from mert_layer_features import LAYERS, WIDTH, LayerMertEncoder
from prepare_dataset import save_json
from prepare_expansion import AUDIT, MANIFEST, ROOT, load_frozen


OUT = ROOT / "outputs/expanded"
PLAN = ROOT / "experiments/expanded_model_plan.json"
FEATURES = OUT / "mert_layers.npy"
FEATURE_META = OUT / "mert_layers.json"
PROGRESS = OUT / "mert_layers_progress.json"


def development_rows():
    manifest = load_frozen()
    rows = [row for row in manifest["tracks"]
            if row["phase_role"] in {"train", "development_validation"}]
    if len(rows) != 1206 or any(row["split"] == "test" for row in rows):
        raise ValueError("Expanded encoder rows crossed the frozen test boundary.")
    audit = json.loads(AUDIT.read_text())
    if audit["manifest_sha256"] != sha256(MANIFEST) or audit["verified_tracks"] != len(rows):
        raise ValueError("Expanded audio audit does not match the manifest.")
    ids = [row["track_id"] for row in rows]
    if set(audit["tracks"]) != set(ids) or audit["new_holdout_audio_verified"] != 0:
        raise ValueError("Audio audit includes the wrong data roles.")
    hashes = [audit["tracks"][track_id]["wav_sha256"] for track_id in ids]
    return rows, hashes


def metadata(device):
    return {
        "encoder_manifest_sha256": sha256(DEST / "SOURCES.json"),
        "base_extractor_sha256": sha256(ROOT / "mert_features.py"),
        "layer_extractor_sha256": sha256(ROOT / "mert_layer_features.py"),
        "driver_source_sha256": sha256(Path(__file__)),
        "expanded_manifest_sha256": sha256(MANIFEST),
        "model_plan_sha256": sha256(PLAN),
        "device": device,
        "versions": {package: version(package) for package in
                     ("torch", "transformers", "librosa", "numpy", "soxr")},
    }


def pilot(device):
    rows, _ = development_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    encoder = LayerMertEncoder(device)
    path = ROOT / rows[0]["audio_file"]
    before = time.perf_counter()
    layers = encoder.extract_layers(path)
    seconds = time.perf_counter() - before
    before = time.perf_counter()
    previous_pooling = encoder.extract(path)
    comparison_seconds = time.perf_counter() - before
    difference = float(np.max(np.abs(layers[1:].mean(axis=0) - previous_pooling)))
    if layers.shape != (LAYERS, WIDTH) or difference > 1e-5:
        raise ValueError("Layer-wise pilot disagrees with the previous transformer-layer mean.")
    result = {
        "track_id": rows[0]["track_id"], "audio_file": rows[0]["audio_file"],
        "shape": list(layers.shape), "finite": bool(np.isfinite(layers).all()),
        "extraction_seconds": seconds, "comparison_seconds": comparison_seconds,
        "max_abs_previous_pooling_difference": difference,
        "python": platform.python_version(), "platform": platform.platform(),
        **metadata(device),
    }
    np.save(OUT / "mert_layers_pilot.npy", layers, allow_pickle=False)
    save_json(OUT / "mert_layers_pilot.json", result)
    print(json.dumps(result, indent=2))


def extract(device):
    rows, hashes = development_rows()
    meta = metadata(device)
    pilot_result = json.loads((OUT / "mert_layers_pilot.json").read_text())
    for key, value in meta.items():
        if pilot_result[key] != value:
            raise ValueError("Pilot does not match this exact extractor/runtime/plan.")
    ids = [row["track_id"] for row in rows]
    if FEATURE_META.exists():
        final = json.loads(FEATURE_META.read_text())
        if final["feature_sha256"] != sha256(FEATURES) or final["ids"] != ids or final["audio_hashes"] != hashes:
            raise ValueError("Completed expanded feature cache is stale.")
        values = np.load(FEATURES, mmap_mode="r", allow_pickle=False)
        if values.shape != (len(rows), LAYERS, WIDTH) or not np.isfinite(values).all():
            raise ValueError("Completed expanded feature matrix is invalid.")
        print("Expanded MERT feature cache already complete and verified.")
        return
    execution = OUT / "mert_layers_execution.json"
    if execution.exists() and json.loads(execution.read_text()) != meta:
        raise ValueError("Expanded extraction was frozen with different settings.")
    save_json(execution, meta)
    completed = 0
    if PROGRESS.exists():
        progress = json.loads(PROGRESS.read_text())
        if progress["metadata"] != meta or progress["ids"] != ids or progress["audio_hashes"] != hashes:
            raise ValueError("Partial expanded feature cache is stale.")
        completed = progress["completed"]
        values = np.lib.format.open_memmap(FEATURES, mode="r+", dtype="float32",
                                           shape=(len(rows), LAYERS, WIDTH))
        if not np.isfinite(values[:completed]).all():
            raise ValueError("Completed portion of feature cache is invalid.")
    else:
        values = np.lib.format.open_memmap(FEATURES, mode="w+", dtype="float32",
                                           shape=(len(rows), LAYERS, WIDTH))
        values[:] = np.nan
        values.flush()
        save_json(PROGRESS, {"metadata": meta, "ids": ids, "audio_hashes": hashes, "completed": 0})
    if not 0 <= completed <= len(rows):
        raise ValueError("Invalid extraction progress count.")
    encoder = LayerMertEncoder(device)
    begin, resumed = time.perf_counter(), completed
    for index, row in enumerate(rows[completed:], completed):
        values[index] = encoder.extract_layers(ROOT / row["audio_file"])
        completed = index + 1
        if completed % 10 == 0 or completed == len(rows):
            values.flush()
            save_json(PROGRESS, {"metadata": meta, "ids": ids, "audio_hashes": hashes,
                                 "completed": completed})
            print(f"Expanded MERT tracks {completed}/{len(rows)}; "
                  f"{time.perf_counter() - begin:.1f}s this run", flush=True)
    del values
    development_rows()
    values = np.load(FEATURES, mmap_mode="r", allow_pickle=False)
    if values.shape != (len(rows), LAYERS, WIDTH) or not np.isfinite(values).all():
        raise ValueError("Final expanded feature matrix is invalid.")
    final = {**meta, "shape": list(values.shape), "dtype": str(values.dtype), "ids": ids,
             "audio_hashes": hashes, "resumed_tracks": resumed, "new_tracks": len(rows) - resumed,
             "extraction_seconds": time.perf_counter() - begin, "test_tracks": 0,
             "feature_sha256": sha256(FEATURES)}
    save_json(FEATURE_META, final)
    print(f"Verified feature cache {FEATURES}: {final['feature_sha256']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pilot", "extract"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    (pilot if args.command == "pilot" else extract)(args.device)


if __name__ == "__main__":
    main()
