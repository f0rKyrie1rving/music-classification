"""Train one fixed MFCC baseline; select thresholds on validation data only."""

import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, extract_features

ROOT = Path(__file__).resolve().parent


def validate_manifest(manifest):
    rows, labels = manifest["tracks"], manifest["labels"]
    audio_hashes = []
    if len({r["track_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate track IDs.")
    artists = {s: {r["artist_id"] for r in rows if r["split"] == s}
               for s in ("train", "validation", "test")}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if artists[a] & artists[b]:
            raise ValueError(f"Artist leakage between {a} and {b}.")
    for row in rows:
        if row["targets"] != [int(label in row["tags"]) for label in labels]:
            raise ValueError("Targets differ from source tags.")
        audio = ROOT / row["audio_file"]
        receipt = ROOT / "data/source/download_receipts" / f'{row["track_id"]}.json'
        if not audio.exists() or not receipt.exists():
            raise ValueError(f'Missing verified audio: {row["track_id"]}; finish data preparation first.')
        digest = hashlib.sha256(audio.read_bytes()).hexdigest()
        if digest != json.loads(receipt.read_text())["wav_sha256"]:
            raise ValueError(f'Audio checksum mismatch: {row["track_id"]}')
        if row.get("wav_sha256", digest) != digest:
            raise ValueError("Audio differs from the frozen manifest.")
        audio_hashes.append(digest)
    for split in artists:
        y = np.array([r["targets"] for r in rows if r["split"] == split])
        if y.ndim != 2 or np.any(y.sum(axis=0) == 0) or np.any(y.sum(axis=0) == len(y)):
            raise ValueError(f"Every label needs positive and negative examples in {split}.")
    return audio_hashes


def choose_thresholds(y, scores):
    """Same fixed grid for each label; ties prefer a threshold near 0.5."""
    candidates = np.linspace(0.1, 0.9, 17)
    return np.array([max(candidates, key=lambda t:
        (f1_score(y[:, j], scores[:, j] >= t, zero_division=0), -abs(t - 0.5)))
        for j in range(y.shape[1])])


def evaluate(y, scores, thresholds, labels):
    prediction = scores >= thresholds
    precision, recall, f1, support = precision_recall_fscore_support(y, prediction, zero_division=0)
    ap = average_precision_score(y, scores, average=None)
    return dict(micro_f1=float(f1_score(y, prediction, average="micro", zero_division=0)),
                macro_f1=float(f1.mean()), macro_ap=float(ap.mean()),
                per_label=[dict(label=label, precision=float(precision[j]), recall=float(recall[j]),
                                f1=float(f1[j]), average_precision=float(ap[j]), support=int(support[j]))
                           for j, label in enumerate(labels)])


def main():
    manifest_path = ROOT / "data/dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audio_hashes = np.array(validate_manifest(manifest))
    rows, labels = manifest["tracks"], manifest["labels"]
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    feature_hash = hashlib.sha256((ROOT / "features.py").read_bytes()).hexdigest()
    ids = np.array([r["track_id"] for r in rows])
    cache = output / "dataset_features.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if (not np.array_equal(data["ids"], ids) or str(data["feature_hash"]) != feature_hash
                    or not np.array_equal(data["audio_hashes"], audio_hashes)):
                raise ValueError("Feature cache is stale; regenerate it explicitly.")
            x = data["x"]
    else:
        vectors = []
        for i, row in enumerate(rows, 1):
            vectors.append(extract_features(ROOT / row["audio_file"]))
            if i % 50 == 0:
                print(f"Extracted {i}/{len(rows)} tracks", flush=True)
        x = np.array(vectors)
        np.savez_compressed(cache, x=x, ids=ids, feature_hash=feature_hash, audio_hashes=audio_hashes)
    if x.shape != (len(rows), len(FEATURE_NAMES)) or not np.isfinite(x).all():
        raise ValueError("Invalid feature matrix.")
    y = np.array([r["targets"] for r in rows])
    masks = {s: np.array([r["split"] == s for r in rows]) for s in ("train", "validation", "test")}
    model = make_pipeline(StandardScaler(), OneVsRestClassifier(
        LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=2026)))
    model.fit(x[masks["train"]], y[masks["train"]])
    prior = y[masks["train"]].mean(axis=0)
    validation_scores = model.predict_proba(x[masks["validation"]])
    thresholds = choose_thresholds(y[masks["validation"]], validation_scores)
    prior_validation = np.tile(prior, (masks["validation"].sum(), 1))
    prior_thresholds = choose_thresholds(y[masks["validation"]], prior_validation)
    results = dict(labels=labels, feature_names=FEATURE_NAMES, feature_source_sha256=feature_hash,
                   manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                   split_counts={s: int(mask.sum()) for s, mask in masks.items()},
                   thresholds=thresholds.tolist(), prior_thresholds=prior_thresholds.tolist(),
                   versions={name: version(name) for name in ("numpy", "librosa", "scikit-learn", "scipy", "joblib")},
                   classifier="StandardScaler + OneVsRest LogisticRegression(C=1, class_weight=balanced)",
                   selection="Fixed model; per-label thresholds selected only on validation Macro-F1 components.")
    for split in ("validation", "test"):
        mask = masks[split]
        scores = model.predict_proba(x[mask])
        results[split] = dict(mfcc=evaluate(y[mask], scores, thresholds, labels),
                             prior=evaluate(y[mask], np.tile(prior, (mask.sum(), 1)), prior_thresholds, labels))
        with (output / f"{split}_predictions.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["track_id", *[f"true_{t}" for t in labels], *[f"score_{t}" for t in labels]])
            for track_id, truth, score in zip(ids[mask], y[mask], scores, strict=True):
                writer.writerow([track_id, *truth, *score])
    (output / "baseline_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    joblib.dump(dict(model=model, thresholds=thresholds, labels=labels, feature_names=FEATURE_NAMES,
                     feature_source_sha256=feature_hash), model_dir / "mfcc_baseline.joblib")
    print(json.dumps({s: results[s] for s in ("validation", "test")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
