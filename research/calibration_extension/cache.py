"""Audit historical MERT caches and extend features to the retired validation cohort.

No fitting or scoring occurs here. The 266-track cohort is retrospective because
its MAEST results were inspected before this extension was designed.
"""

import argparse
import hashlib
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/calibration_extension_cache/20260927_v1"
VALIDATION = ROOT / "outputs/calibration_validation/20260927_v1"
MODELS = {
    "mert_v0": {
        "cache": "outputs/expanded/mert_layers.npy",
        "meta": "outputs/expanded/mert_layers.json",
        "plan": "experiments/expanded_model_plan.json",
        "driver": "extract_expanded_mert.py",
        "prepare": "prepare_mert",
        "encoder": "mert_layer_features",
        "class": "LayerMertEncoder",
        "sources": {"base_extractor_sha256": "mert_features.py",
                    "layer_extractor_sha256": "mert_layer_features.py"},
    },
    "mert_v1": {
        "cache": "outputs/mert_v1/mert_layers.npy",
        "meta": "outputs/mert_v1/mert_layers.json",
        "plan": "experiments/mert_v1_plan_v2.json",
        "driver": "extract_expanded_mert_v1.py",
        "prepare": "prepare_mert_v1",
        "encoder": "mert_v1_features",
        "class": "V1MertEncoder",
        "sources": {"extractor_source_sha256": "mert_v1_features.py"},
    },
}
SCOPE = "Retrospective extension of previously scored 266 tracks; not a fresh confirmatory test."
PACKAGES = ("torch", "transformers", "librosa", "numpy", "soxr", "soundfile")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(2 ** 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def immutable_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen file differs: {path}")
    else:
        save_json(path, value)


def unique_ids(ids):
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(x, str) for x in ids):
        raise ValueError("Track IDs must be nonempty, unique strings.")


def mean_transformer_layers(layers):
    """Match historical float32 storage, then np.mean with float32 accumulation."""
    layers = np.asarray(layers, dtype=np.float32)
    if layers.shape != (13, 768) or not np.isfinite(layers).all():
        raise ValueError("Expected 13 finite layers of width 768.")
    return layers[1:].mean(axis=0)


def validate_arrays(x, ids, expected_ids, completed=None):
    unique_ids(ids)
    if ids != expected_ids:
        raise ValueError("Cache IDs/order differ from the frozen cohort.")
    if x.shape != (len(ids), 768) or x.dtype != np.float32:
        raise ValueError("Feature cache must have float32 shape (tracks, 768).")
    completed = len(ids) if completed is None else completed
    if not isinstance(completed, int) or not 0 <= completed <= len(ids):
        raise ValueError("Invalid cache progress count.")
    if not np.isfinite(x[:completed]).all():
        raise ValueError("Completed feature rows contain non-finite values.")


def runtime():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {package: version(package) for package in PACKAGES},
            "device": "cpu", "torch_threads": 4, "torch_seed": 2026}


def audit():
    """Rehash all original audio, cache files, extraction sources and model files."""
    manifest_path = ROOT / "data/expanded_manifest.json"
    manifest = read_json(manifest_path)
    historical = manifest["tracks"]
    by_id = {row["track_id"]: row for row in historical}
    unique_ids(list(by_id))
    if len(by_id) != len(historical):
        raise ValueError("Duplicate historical tracks.")
    rows = [row for row in historical if row["phase_role"] in ("train", "development_validation")]
    if len(rows) != 1206 or any(row["split"] == "test" for row in rows):
        raise ValueError("Historical development boundary changed.")
    audio_audit_path = ROOT / "data/expanded_development_audit.json"
    audio_audit = read_json(audio_audit_path)
    if (audio_audit["manifest_sha256"] != sha256(manifest_path)
            or set(audio_audit["tracks"]) != {row["track_id"] for row in rows}
            or audio_audit["verified_tracks"] != 1206):
        raise ValueError("Historical audio receipt does not match development rows.")
    old_hashes = {}
    for row in rows:
        digest = sha256(ROOT / row["audio_file"])
        if digest != audio_audit["tracks"][row["track_id"]]["wav_sha256"]:
            raise ValueError(f"Historical audio changed: {row['track_id']}")
        old_hashes[row["track_id"]] = digest
    result = {"scope": SCOPE, "status": "passed", "historical_tracks": len(rows),
              "historical_artists": len({row["artist_id"] for row in rows}),
              "historical_manifest_sha256": sha256(manifest_path),
              "historical_audio_audit_sha256": sha256(audio_audit_path),
              "runtime": runtime(), "models": {}}
    verify = importlib.import_module("prepare_mert").verify
    for name, spec in MODELS.items():
        meta_path, cache_path = ROOT / spec["meta"], ROOT / spec["cache"]
        meta = read_json(meta_path)
        plan_path = ROOT / spec["plan"]
        plan = read_json(plan_path)
        ids = ([row["track_id"] for row in rows] if name == "mert_v0"
               else plan["training_ids"] + plan["validation_ids"])
        unique_ids(ids)
        if set(ids) != set(old_hashes) or meta["ids"] != ids:
            raise ValueError(f"Historical cache IDs mismatch: {name}")
        if (meta["audio_hashes"] != [old_hashes[track] for track in ids]
                or meta["feature_sha256"] != sha256(cache_path)):
            raise ValueError(f"Historical feature/audio digest mismatch: {name}")
        checks = {**spec["sources"], "driver_source_sha256": spec["driver"],
                  "expanded_manifest_sha256": "data/expanded_manifest.json",
                  "model_plan_sha256": spec["plan"]}
        for key, path in checks.items():
            if sha256(ROOT / path) != meta[key]:
                raise ValueError(f"Historical source changed: {path}")
        values = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        if (values.shape != (1206, 13, 768) or values.dtype != np.float32
                or not np.isfinite(values).all() or meta["device"] != "cpu"):
            raise ValueError(f"Invalid historical cache geometry/device: {name}")
        for package, v in meta["versions"].items():
            if version(package) != v:
                raise ValueError(f"Historical runtime differs: {package}")
        prepare = importlib.import_module(spec["prepare"])
        model_receipt = prepare.DEST / "SOURCES.json"
        source = read_json(model_receipt)
        if (sha256(model_receipt) != meta["encoder_manifest_sha256"]
                or source["revision"] != prepare.REVISION
                or set(source["files_sha256"]) != set(prepare.FILES)):
            raise ValueError(f"Model source receipt mismatch: {name}")
        for file, expected in prepare.FILES.items():
            if verify(prepare.DEST / file, expected) != source["files_sha256"][file]:
                raise ValueError(f"Pinned model changed: {name}/{file}")
        result["models"][name] = {
            "cache_sha256": sha256(cache_path), "metadata_sha256": sha256(meta_path),
            "ids": ids, "audio_hashes": [old_hashes[track] for track in ids],
            "model_receipt_sha256": sha256(model_receipt),
            "model_id": source["model_id"], "revision": source["revision"],
            "model_files_sha256": source["files_sha256"],
            "sources_sha256": {path: sha256(ROOT / path) for path in checks.values()},
            "shape": list(values.shape), "dtype": str(values.dtype),
        }
    new_rows = read_json(VALIDATION / "manifest.json")["tracks"]
    ids = [row["track_id"] for row in new_rows]
    unique_ids(ids)
    receipts = read_json(VALIDATION / "download_status.json")
    receipt_by_id = {row["track_id"]: row for row in receipts["successful"]}
    if (len(new_rows) != 266 or len(receipt_by_id) != 266 or receipts["failures"]
            or set(receipt_by_id) != set(ids)):
        raise ValueError("Retrospective acquisition cohort is not complete.")
    old_artists = {row["artist_id"] for row in historical}
    new_artists = {row["artist_id"] for row in new_rows}
    if set(ids) & set(by_id) or old_artists & new_artists or len(new_artists) != 200:
        raise ValueError("Retrospective cohort crosses the historical artist boundary.")
    new_hashes = [sha256(ROOT / row["audio_file"]) for row in new_rows]
    if new_hashes != [receipt_by_id[track]["wav_sha256"] for track in ids]:
        raise ValueError("Retrospective WAV hashes differ from acquisition receipts.")
    result["retrospective"] = {
        "ids": ids, "audio_hashes": new_hashes, "tracks": 266, "artists": 200,
        "manifest_sha256": sha256(VALIDATION / "manifest.json"),
        "download_status_sha256": sha256(VALIDATION / "download_status.json"),
        "scope": SCOPE,
    }
    immutable_json(OUT / "audit.json", result)
    return result


def freeze(audited):
    sources = [Path(__file__), ROOT / "research/calibration_extension/test_cache.py",
               ROOT / "prepare_mert.py", ROOT / "prepare_mert_v1.py",
               ROOT / "mert_features.py", ROOT / "mert_layer_features.py",
               ROOT / "mert_v1_features.py"]
    value = {"audit_sha256": sha256(OUT / "audit.json"), "scope": SCOPE,
             "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
             "runtime": audited["runtime"], "device": "cpu", "checkpoint_tracks": 25,
             "method": "Historical extract_layers -> float32 13x768 -> layers[1:].mean(axis=0), float32 accumulation",
             "control_absolute_tolerance": 1e-5, "no_model_download": True,
             "no_training_or_prediction": True, "models": list(MODELS),
             "ids": audited["retrospective"]["ids"],
             "audio_hashes": audited["retrospective"]["audio_hashes"]}
    immutable_json(OUT / "freeze.json", value)
    for source in sources:
        destination = OUT / "source_snapshot" / source.relative_to(ROOT)
        if destination.exists():
            if sha256(destination) != sha256(source):
                raise ValueError(f"Source snapshot differs: {source}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
    return value


def complete_cache(path, receipt_path, expected_ids, freeze_sha):
    if not receipt_path.exists():
        if path.exists():
            raise ValueError("Final cache has no completion receipt; inspect before adopting it.")
        return False
    receipt = read_json(receipt_path)
    if receipt["freeze_sha256"] != freeze_sha or sha256(path) != receipt["feature_sha256"]:
        raise ValueError("Completed cache or frozen protocol changed.")
    with np.load(path, allow_pickle=False) as archive:
        validate_arrays(archive["x"], archive["ids"].tolist(), expected_ids)
    return True


def extract(name, audited):
    spec = MODELS[name]
    ids, hashes = (audited["retrospective"][key] for key in ("ids", "audio_hashes"))
    frozen_sha = sha256(OUT / "freeze.json")
    path, receipt_path = OUT / f"{name}_features.npz", OUT / f"{name}_receipt.json"
    if complete_cache(path, receipt_path, ids, frozen_sha):
        print(f"{name}: completed cache verified; no overwrite.", flush=True)
        return
    rows = read_json(VALIDATION / "manifest.json")["tracks"]
    module = importlib.import_module(spec["encoder"])
    encoder = getattr(module, spec["class"])("cpu")
    first_id = audited["models"][name]["ids"][0]
    historical = read_json(ROOT / "data/expanded_manifest.json")["tracks"]
    control_row = next(row for row in historical if row["track_id"] == first_id)
    start = time.perf_counter()
    control = mean_transformer_layers(encoder.extract_layers(ROOT / control_row["audio_file"]))
    old = np.load(ROOT / spec["cache"], mmap_mode="r", allow_pickle=False)
    expected = old[0, 1:].mean(axis=0)
    difference = float(np.max(np.abs(control - expected)))
    if difference > 1e-5:
        raise ValueError(f"{name}: control differs from old cache by {difference}")
    control_receipt = {"track_id": first_id, "audio_sha256": sha256(ROOT / control_row["audio_file"]),
                       "max_abs_difference": difference, "exact_equal": bool(np.array_equal(control, expected)),
                       "absolute_tolerance": 1e-5, "freeze_sha256": frozen_sha}
    immutable_json(OUT / f"{name}_control.json", control_receipt)
    print(f"{name}: historical control max_abs_difference={difference}; {time.perf_counter()-start:.1f}s", flush=True)
    partial_path, progress_path = OUT / f"{name}_partial.npy", OUT / f"{name}_progress.json"
    if progress_path.exists():
        progress = read_json(progress_path)
        if (progress["ids"] != ids or progress["audio_hashes"] != hashes
                or progress["freeze_sha256"] != frozen_sha
                or progress["partial_sha256"] != sha256(partial_path)):
            raise ValueError(f"{name}: stale or modified partial cache.")
        values = np.load(partial_path, allow_pickle=False)
        completed = progress["completed"]
        validate_arrays(values, ids, ids, completed)
    else:
        if partial_path.exists():
            raise ValueError(f"{name}: partial feature file has no progress receipt.")
        values = np.full((len(ids), 768), np.nan, dtype=np.float32)
        completed = 0
    resumed = completed
    for i in range(completed, len(rows)):
        audio_path = ROOT / rows[i]["audio_file"]
        if sha256(audio_path) != hashes[i]:
            raise ValueError(f"Audio changed during extraction: {ids[i]}")
        values[i] = mean_transformer_layers(encoder.extract_layers(audio_path))
        completed = i + 1
        if completed % 25 == 0 or completed == len(rows):
            temporary = partial_path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
            temporary.replace(partial_path)
            save_json(progress_path, {"completed": completed, "ids": ids, "audio_hashes": hashes,
                                     "freeze_sha256": frozen_sha, "partial_sha256": sha256(partial_path)})
            print(f"{name}: {completed}/{len(rows)} tracks, {time.perf_counter()-start:.1f}s this run", flush=True)
    validate_arrays(values, ids, ids)
    # Verify frozen source and audio identities again before publishing the completed cache.
    for source, expected_hash in read_json(OUT / "freeze.json")["source_sha256"].items():
        if sha256(ROOT / source) != expected_hash:
            raise ValueError(f"Source changed during extraction: {source}")
    if [sha256(ROOT / row["audio_file"]) for row in rows] != hashes:
        raise ValueError("Audio changed during extraction.")
    with path.open("xb") as stream:
        np.savez_compressed(stream, x=values, ids=np.asarray(ids))
    receipt = {"status": "complete", "scope": SCOPE, "encoder": name,
               "shape": list(values.shape), "dtype": str(values.dtype), "ids": ids,
               "audio_hashes": hashes, "feature_sha256": sha256(path),
               "freeze_sha256": frozen_sha, "control": control_receipt,
               "extraction_seconds": time.perf_counter()-start, "resumed_tracks": resumed,
               "new_tracks": len(ids)-resumed, "failures": [], "replacement_tracks": 0}
    save_json(receipt_path, receipt)
    complete_cache(path, receipt_path, ids, frozen_sha)
    print(f"{name}: complete {path.name}; SHA256 {receipt['feature_sha256']}", flush=True)


def main():
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    os.environ["NUMBA_CACHE_DIR"] = str(OUT / "runtime_cache/numba")
    os.environ["HF_HOME"] = str(OUT / "runtime_cache/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "extract"))
    parser.add_argument("--encoder", choices=("all", *MODELS), default="all")
    args = parser.parse_args()
    audited = audit()
    print("Audited 1206 historical and 266 retrospective tracks; both MERT caches and pinned models verified.", flush=True)
    if args.command == "extract":
        freeze(audited)
        for name in MODELS if args.encoder == "all" else [args.encoder]:
            extract(name, audited)


if __name__ == "__main__":
    main()
