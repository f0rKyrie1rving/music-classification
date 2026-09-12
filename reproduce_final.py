"""Replay the fixed final model in separate outputs, without rewriting its frozen plan.

This is computational reproduction of an already observed evaluation, not a new
test or a rerun of all historical representation-selection experiments.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_release import ROOT, assert_matches, check_baseline, evaluate, load_release, read_json, sha256

OUT = ROOT / "outputs/reproduction"
TELEMETRY = {"extraction_seconds", "resumed_tracks", "new_tracks"}


def stable_metadata(meta):
    """Historical extraction telemetry is not scientific cache identity."""
    return {key: value for key, value in meta.items() if key not in TELEMETRY}


def inputs():
    check_baseline()
    from prepare_expansion import load_frozen
    manifest = load_frozen()  # Includes the now-published exact pool/index hashes.
    plan, _, published, _, _ = load_release()
    by_id = {r["track_id"]: r for r in manifest["tracks"]}
    audits = {"development": read_json(ROOT / "data/expanded_development_audit.json"),
              "holdout": read_json(ROOT / "data/final_holdout_audit.json")}
    rows = {"development": [by_id[i] for i in plan["fit_ids"]],
            "holdout": [by_id[i] for i in plan["holdout_ids"]]}
    for role, selected in rows.items():
        if audits[role]["manifest_sha256"] != sha256(ROOT / "data/expanded_manifest.json"):
            raise ValueError("Archived audio audit differs from the manifest.")
        if set(audits[role]["tracks"]) != {r["track_id"] for r in selected}:
            raise ValueError("Archived audio audit has the wrong track IDs.")
    return plan, rows, audits, published


def verify_audio(row, audit):
    path = ROOT / row["audio_file"]
    if sha256(path) != audit["tracks"][row["track_id"]]["wav_sha256"]:
        raise ValueError(f"Reconstructed WAV differs from the archived excerpt: {row['track_id']}")


def download(workers):
    from prepare_dataset import download_track
    from repair_short_audio import repair
    _, rows, audits, _ = inputs()

    def acquire(item):
        row, audit = item
        path = ROOT / row["audio_file"]
        if path.exists():
            verify_audio(row, audit)
            return  # Never replace existing audio that differs from its archived hash.
        try:
            download_track(row)
        except ValueError as error:
            if "Not enough decodable audio within bounded prefix" not in str(error):
                raise
            repair(row)
        verify_audio(row, audit)

    jobs = [(row, audits[role]) for role, selected in rows.items() for row in selected]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, _ in enumerate(pool.map(acquire, jobs), 1):
            if count % 100 == 0 or count == len(jobs):
                print(f"Verified audio: {count}/{len(jobs)}", flush=True)


def provenance(role, rows, audit, device):
    return dict(role=role, ids=[r["track_id"] for r in rows],
                audio_hashes=[audit["tracks"][r["track_id"]]["wav_sha256"] for r in rows],
                plan_sha256=sha256(ROOT / "experiments/final_holdout_plan.json"),
                extractor_sha256=sha256(ROOT / "maest_hf_features.py"),
                driver_sha256=sha256(Path(__file__)),
                model_receipt_sha256=sha256(ROOT / "models/mtg-upf-maest-519l/SOURCES.json"),
                runtime_lock_sha256=sha256(ROOT / "requirements-mert-lock.txt"),
                device=device, versions={p: version(p) for p in
                    ("numpy", "torch", "transformers", "safetensors", "soundfile", "soxr")})


def validate_features(values, meta, expected, digest):
    assert_matches(meta["provenance"], expected, "feature provenance")
    if (meta["feature_sha256"] != digest or values.shape != (len(expected["ids"]), 2304)
            or values.dtype != np.float32 or not np.isfinite(values).all()):
        raise ValueError("Feature cache checksum, shape, dtype or values are invalid.")


def extract(device):
    from maest_hf_features import HfMaestEncoder
    from prepare_dataset import save_json
    plan, rows, audits, _ = inputs()
    if sha256(ROOT / "models/mtg-upf-maest-519l/SOURCES.json") != plan["model_receipt_sha256"]:
        raise ValueError("MAEST receipt differs from the frozen model.")
    for role, selected in rows.items():
        for row in selected:
            verify_audio(row, audits[role])
    encoder = None
    OUT.mkdir(parents=True, exist_ok=True)
    for role, selected in rows.items():
        expected = provenance(role, selected, audits[role], device)
        array_path, meta_path = OUT / f"{role}.npy", OUT / f"{role}.json"
        progress_path = OUT / f"{role}_progress.json"
        if meta_path.exists():
            values = np.load(array_path, allow_pickle=False, mmap_mode="r")
            validate_features(values, read_json(meta_path), expected, sha256(array_path))
            print(f"Verified completed {role} cache.", flush=True)
            continue
        if progress_path.exists():
            progress = read_json(progress_path)
            assert_matches(progress["provenance"], expected)
            completed = progress["completed"]
            values = np.load(array_path, allow_pickle=False, mmap_mode="r+")
            if (not 0 <= completed <= len(selected) or values.shape != (len(selected), 2304)
                    or values.dtype != np.float32 or not np.isfinite(values[:completed]).all()):
                raise ValueError("Partial feature cache is invalid.")
        else:
            if array_path.exists():
                raise FileExistsError(f"Unidentified cache exists; inspect it before continuing: {array_path}")
            values = np.lib.format.open_memmap(array_path, mode="w+", dtype="float32", shape=(len(selected), 2304))
            values[:] = np.nan
            values.flush()
            completed = 0
            save_json(progress_path, dict(provenance=expected, completed=0))
        if encoder is None:
            encoder = HfMaestEncoder(device)
        start, resumed = time.perf_counter(), completed
        for index in range(completed, len(selected)):
            values[index] = encoder.extract(ROOT / selected[index]["audio_file"])
            if (index + 1) % 10 == 0 or index + 1 == len(selected):
                values.flush()
                save_json(progress_path, dict(provenance=expected, completed=index + 1))
                print(f"{role}: {index + 1}/{len(selected)}", flush=True)
        del values
        meta = dict(provenance=expected, feature_sha256=sha256(array_path))
        validate_features(np.load(array_path, allow_pickle=False), meta, expected, sha256(array_path))
        # Only stable provenance and the feature digest go in the scientific record.
        save_json(meta_path, meta)
        save_json(OUT / f"{role}_runtime.json", dict(extraction_seconds=time.perf_counter() - start,
                  resumed_tracks=resumed, new_tracks=len(selected) - resumed))


def retained_features(plan):
    meta = read_json(ROOT / "outputs/maest_hf/features.json")
    expected = read_json(ROOT / "data/evaluation/development_feature_provenance.json")
    if stable_metadata(meta) != expected:
        raise ValueError("Retained development feature provenance changed.")
    paths = (ROOT / "outputs/maest_hf/features.npy", ROOT / "outputs/final_maest/holdout_features.npy")
    reference = read_json(ROOT / "data/evaluation/original_metrics.json")
    digests = (plan["development_features_sha256"], reference["holdout_feature_sha256"])
    values = []
    for path, digest, count in zip(paths, digests, (len(plan["fit_ids"]), len(plan["holdout_ids"])), strict=True):
        if sha256(path) != digest:
            raise ValueError(f"Retained feature bytes changed: {path}")
        array = np.load(path, allow_pickle=False)
        if array.shape != (count, 2304) or not np.isfinite(array).all():
            raise ValueError("Invalid retained features.")
        values.append(array)
    return values


def run(device, retained=False):
    plan, rows, audits, published = inputs()
    result_path = OUT / ("retained_replay.json" if retained else "fresh_replay.json")
    if result_path.exists():
        raise FileExistsError(f"Reproduction result already exists: {result_path}")
    if retained:
        x_fit, x_hold = retained_features(plan)
    else:
        arrays = []
        for role, selected in rows.items():
            path = OUT / f"{role}.npy"
            array = np.load(path, allow_pickle=False)
            validate_features(array, read_json(OUT / f"{role}.json"),
                              provenance(role, selected, audits[role], device), sha256(path))
            arrays.append(array)
        x_fit, x_hold = arrays
    y_fit = np.array([[int(bool(set(row["tags"]) & set(plan["ontology"][label])))
                       for label in plan["labels"]] for row in rows["development"]])
    scores = []
    for j, item in enumerate(plan["selected"]):
        if item["label"] != plan["labels"][j]:
            raise ValueError("Selected head order changed.")
        model = make_pipeline(StandardScaler(), LogisticRegression(**item["config"], max_iter=3000, random_state=2026))
        model.fit(x_fit, y_fit[:, j])
        scores.append(model.predict_proba(x_hold)[:, 1])
    scores = np.column_stack(scores)
    targets = np.array([r["targets"] for r in published])
    original = np.array([r["scores"] for r in published])
    result = dict(scope="computational replay of fixed final heads; not new independent evidence",
                  retained_features=retained, plan_sha256=sha256(ROOT / "experiments/final_holdout_plan.json"),
                  runner_sha256=sha256(Path(__file__)),
                  versions={p: version(p) for p in ("numpy", "scikit-learn", "scipy")},
                  max_absolute_score_difference=float(np.max(np.abs(scores - original))),
                  decision_mismatches={policy: int(np.count_nonzero(
                      (scores >= np.array(t)) != (original >= np.array(t)))) for policy, t in plan["thresholds"].items()},
                  policies={policy: evaluate(targets, scores, t) for policy, t in plan["thresholds"].items()},
                  ids=plan["holdout_ids"], targets=targets.tolist(), scores=scores.tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    with result_path.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps({k: result[k] for k in ("scope", "max_absolute_score_difference", "decision_mismatches")}, indent=2))
    print(f"Saved {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "download", "extract", "run"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--retained-features", action="store_true", help="Local archived-cache replay, run only")
    args = parser.parse_args()
    if args.workers < 1 or (args.retained_features and args.command != "run"):
        parser.error("workers must be positive; --retained-features is only valid with run")
    if args.command == "check":
        inputs()
        print("Verified published pool, indexes, plans, manifests, archived audits and score rows.")
    elif args.command == "download":
        download(args.workers)
    elif args.command == "extract":
        extract(args.device)
    else:
        run(args.device, args.retained_features)
