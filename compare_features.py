"""Run the frozen 26-vs-46 feature comparison without predicting test audio."""

import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from extra_features import EXTENDED_NAMES, extract_extended_features
from train import choose_thresholds, evaluate

ROOT = Path(__file__).resolve().parent
PLAN_PATH = ROOT / "experiments/feature_comparison_plan.json"
OUTPUT = ROOT / "outputs/feature_comparison"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_protected(plan):
    for name, expected in plan["protected_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Protected baseline changed: {name}")
    for name, key in [("extra_features.py", "additional_features_sha256"),
                      ("experiments/PROTOCOL.md", "protocol_sha256")]:
        if digest(ROOT / name) != plan[key]:
            raise ValueError(f"Frozen experiment definition changed: {name}")


def development_rows(manifest):
    rows = manifest["tracks"]
    if len({r["track_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate track IDs.")
    artists = {s: {r["artist_id"] for r in rows if r["split"] == s}
               for s in ("train", "validation", "test")}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if artists[a] & artists[b]:
            raise ValueError("Artist leakage across partitions.")
    return [r for r in rows if r["split"] in ("train", "validation")]


def counts(y, scores, thresholds):
    predicted = scores >= thresholds
    return [dict(tp=int((predicted[:, j] & (y[:, j] == 1)).sum()),
                 fp=int((predicted[:, j] & (y[:, j] == 0)).sum()),
                 fn=int((~predicted[:, j] & (y[:, j] == 1)).sum()),
                 tn=int((~predicted[:, j] & (y[:, j] == 0)).sum()))
            for j in range(y.shape[1])]


def main():
    plan = json.loads(PLAN_PATH.read_text())
    check_protected(plan)
    if plan["feature_sets"]["extended"] != EXTENDED_NAMES:
        raise ValueError("Feature order differs from the frozen plan.")
    manifest = json.loads((ROOT / "data/dataset_manifest.json").read_text())
    rows, labels = development_rows(manifest), manifest["labels"]
    train_mask = np.array([r["split"] == "train" for r in rows])
    valid_mask = ~train_mask
    if (int(train_mask.sum()), int(valid_mask.sum()), labels) != (300, 90, plan["labels"]):
        raise ValueError("Development split differs from the fixed plan.")
    for row in rows:
        if row["targets"] != [int(t in row["tags"]) for t in labels]:
            raise ValueError("Target differs from original annotations.")
        if digest(ROOT / row["audio_file"]) != row["wav_sha256"]:
            raise ValueError(f'Audio hash mismatch: {row["track_id"]}')
    ids = np.array([r["track_id"] for r in rows])
    with np.load(ROOT / "outputs/dataset_features.npz", allow_pickle=False) as cached:
        positions = {track_id: i for i, track_id in enumerate(cached["ids"])}
        baseline_x = cached["x"][[positions[track_id] for track_id in ids]]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache_path, plan_hash = OUTPUT / "development_features.npz", digest(PLAN_PATH)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["plan_sha256"]) != plan_hash or not np.array_equal(cached["ids"], ids):
                raise ValueError("Comparison cache differs from the frozen plan.")
            extended_x = cached["x"]
    else:
        vectors = []
        for i, row in enumerate(rows, 1):
            vectors.append(extract_extended_features(ROOT / row["audio_file"]))
            if i % 50 == 0:
                print(f"Extended features: {i}/{len(rows)} development tracks", flush=True)
        extended_x = np.array(vectors)
        np.savez_compressed(cache_path, x=extended_x, ids=ids, plan_sha256=plan_hash)
    if extended_x.shape != (390, 46) or not np.isfinite(extended_x).all():
        raise ValueError("Unexpected comparison matrix.")
    if not np.array_equal(extended_x[:, :26], baseline_x):
        raise ValueError("Candidate MFCCs differ from the original baseline.")
    y = np.array([r["targets"] for r in rows])
    report = dict(plan_sha256=plan_hash, train_count=300, validation_count=90,
                  test_prediction_count=0, labels=labels, primary_metric=plan["primary_metric"],
                  versions={name: version(name) for name in ("librosa", "numpy", "scikit-learn", "scipy")},
                  models={})
    for name, x in [("mfcc", baseline_x), ("extended", extended_x)]:
        model = make_pipeline(StandardScaler(), OneVsRestClassifier(LogisticRegression(**plan["model"])))
        model.fit(x[train_mask], y[train_mask])
        np.testing.assert_allclose(model[0].mean_, x[train_mask].mean(axis=0), rtol=0, atol=1e-12)
        scores = model.predict_proba(x[valid_mask])
        thresholds = choose_thresholds(y[valid_mask], scores)
        result = dict(n_features=x.shape[1], thresholds=thresholds.tolist(),
                      validation=evaluate(y[valid_mask], scores, thresholds, labels),
                      fixed_05=evaluate(y[valid_mask], scores, np.full(4, 0.5), labels),
                      counts=counts(y[valid_mask], scores, thresholds))
        if name == "mfcc":
            old = json.loads((ROOT / "outputs/baseline_metrics.json").read_text())
            if result["validation"] != old["validation"]["mfcc"] or result["thresholds"] != old["thresholds"]:
                raise ValueError("Reproduced baseline differs from the earlier run.")
        report["models"][name] = result
        with (OUTPUT / f"{name}_validation_predictions.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["track_id", *[f"true_{t}" for t in labels], *[f"score_{t}" for t in labels]])
            for track_id, truth, score in zip(ids[valid_mask], y[valid_mask], scores, strict=True):
                writer.writerow([track_id, *truth, *score])
        # Save experimental artifacts separately; predict.py continues using the old model.
        joblib.dump(dict(model=model, thresholds=thresholds, labels=labels,
                         feature_names=plan["feature_sets"][name], plan_sha256=plan_hash,
                         evaluation_scope="validation-only; not the default demo"), OUTPUT / f"{name}.joblib")
    check_protected(plan)
    report["baseline_artifacts_unchanged"] = True
    (OUTPUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: result["validation"] for name, result in report["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
