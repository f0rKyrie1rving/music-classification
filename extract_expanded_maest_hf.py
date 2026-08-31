"""Pilot and resumable official Hugging Face MAEST development extraction."""

import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from maest_hf_features import DEST, FEATURE_WIDTH, HfMaestEncoder
from mert_features import sha256
from prepare_dataset import save_json
from prepare_expansion import AUDIT, MANIFEST, ROOT, load_frozen


OUT = ROOT / "outputs/maest_hf"
PLAN = ROOT / "experiments/maest_hf_plan.json"
FEATURES = OUT / "features.npy"
FEATURE_META = OUT / "features.json"
PROGRESS = OUT / "features_progress.json"


def development_rows():
    manifest, plan = load_frozen(), json.loads(PLAN.read_text())
    by_id = {row["track_id"]: row for row in manifest["tracks"]}
    ids = plan["training_ids"] + plan["validation_ids"]
    if set(ids) & set(plan["excluded_test_ids"]):
        raise ValueError("Test ID entered the MAEST feature plan.")
    rows = [by_id[track_id] for track_id in ids]
    if len(rows) != 1206 or any(row["split"] == "test" for row in rows):
        raise ValueError("MAEST encoder rows crossed the test boundary.")
    audit = json.loads(AUDIT.read_text())
    if audit["manifest_sha256"] != sha256(MANIFEST) or set(audit["tracks"]) != set(ids):
        raise ValueError("Expanded audio audit differs from MAEST rows.")
    hashes = [audit["tracks"][track_id]["wav_sha256"] for track_id in ids]
    return rows, hashes


def metadata(device):
    return {"encoder_manifest_sha256": sha256(DEST / "SOURCES.json"),
            "extractor_source_sha256": sha256(ROOT / "maest_hf_features.py"),
            "driver_source_sha256": sha256(Path(__file__)),
            "expanded_manifest_sha256": sha256(MANIFEST),
            "model_plan_sha256": sha256(PLAN), "device": device,
            "versions": {package: version(package) for package in
                         ("torch", "transformers", "safetensors", "numpy", "soundfile", "soxr")}}


def pilot(device):
    rows, _ = development_rows(); OUT.mkdir(parents=True, exist_ok=True)
    encoder = HfMaestEncoder(device)
    path = ROOT / rows[0]["audio_file"]
    before = time.perf_counter(); first = encoder.extract(path)
    first_seconds = time.perf_counter() - before
    before = time.perf_counter(); second = encoder.extract(path)
    second_seconds = time.perf_counter() - before
    difference = float(np.max(np.abs(first - second)))
    if first.shape != (FEATURE_WIDTH,) or not np.isfinite(first).all() or difference > 1e-5:
        raise ValueError("MAEST pilot is invalid or non-repeatable.")
    result = {"track_id": rows[0]["track_id"], "audio_file": rows[0]["audio_file"],
              "shape": list(first.shape), "finite": True,
              "first_seconds": first_seconds, "second_seconds": second_seconds,
              "max_abs_repeat_difference": difference, "python": platform.python_version(),
              "platform": platform.platform(), "encoder_training": encoder.model.training,
              "parameters_require_grad": any(p.requires_grad for p in encoder.model.parameters()),
              "test_tracks": 0, **metadata(device)}
    np.save(OUT / "pilot.npy", first, allow_pickle=False)
    save_json(OUT / "pilot.json", result); print(json.dumps(result, indent=2))


def extract(device):
    rows, hashes = development_rows(); meta = metadata(device)
    pilot_result = json.loads((OUT / "pilot.json").read_text())
    for key, value in meta.items():
        if pilot_result[key] != value:
            raise ValueError("MAEST pilot differs from this runtime and plan.")
    ids = [row["track_id"] for row in rows]
    if FEATURE_META.exists():
        final = json.loads(FEATURE_META.read_text())
        values = np.load(FEATURES, mmap_mode="r", allow_pickle=False)
        if (final["feature_sha256"] != sha256(FEATURES) or final["ids"] != ids
                or final["audio_hashes"] != hashes or values.shape != (len(rows), FEATURE_WIDTH)
                or not np.isfinite(values).all()):
            raise ValueError("Completed MAEST feature cache is stale or invalid.")
        print("MAEST feature cache already complete and verified."); return
    execution = OUT / "execution.json"
    if execution.exists() and json.loads(execution.read_text()) != meta:
        raise ValueError("MAEST extraction was frozen with different settings.")
    save_json(execution, meta); completed = 0
    if PROGRESS.exists():
        progress = json.loads(PROGRESS.read_text())
        if progress["metadata"] != meta or progress["ids"] != ids or progress["audio_hashes"] != hashes:
            raise ValueError("Partial MAEST feature cache is stale.")
        completed = progress["completed"]
        values = np.lib.format.open_memmap(
            FEATURES, mode="r+", dtype="float32", shape=(len(rows), FEATURE_WIDTH))
        if not np.isfinite(values[:completed]).all():
            raise ValueError("Completed MAEST features are invalid.")
    else:
        values = np.lib.format.open_memmap(
            FEATURES, mode="w+", dtype="float32", shape=(len(rows), FEATURE_WIDTH))
        values[:] = np.nan; values.flush()
        save_json(PROGRESS, {"metadata": meta, "ids": ids, "audio_hashes": hashes, "completed": 0})
    encoder = HfMaestEncoder(device); begin, resumed = time.perf_counter(), completed
    for index, row in enumerate(rows[completed:], completed):
        values[index] = encoder.extract(ROOT / row["audio_file"]); completed = index + 1
        if completed % 10 == 0 or completed == len(rows):
            values.flush()
            save_json(PROGRESS, {"metadata": meta, "ids": ids, "audio_hashes": hashes,
                                 "completed": completed})
            print(f"MAEST tracks {completed}/{len(rows)}; {time.perf_counter()-begin:.1f}s this run",
                  flush=True)
    del values; development_rows()
    values = np.load(FEATURES, mmap_mode="r", allow_pickle=False)
    if values.shape != (len(rows), FEATURE_WIDTH) or not np.isfinite(values).all():
        raise ValueError("Final MAEST features are invalid.")
    final = {**meta, "shape": list(values.shape), "dtype": str(values.dtype), "ids": ids,
             "audio_hashes": hashes, "resumed_tracks": resumed, "new_tracks": len(rows)-resumed,
             "extraction_seconds": time.perf_counter()-begin, "test_tracks": 0,
             "feature_sha256": sha256(FEATURES)}
    save_json(FEATURE_META, final)
    print(f"Verified MAEST feature cache: {final['feature_sha256']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pilot", "extract"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args(); (pilot if args.command == "pilot" else extract)(args.device)
