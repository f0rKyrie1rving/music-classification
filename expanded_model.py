"""Freeze and run the expanded broad-genre development experiment."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prepare_dataset import save_json
from prepare_expansion import AUDIT, MANIFEST, ROOT, load_frozen


OUT = ROOT / "outputs/expanded"
PLAN = ROOT / "experiments/expanded_model_plan.json"
PROTOCOL = ROOT / "experiments/EXPANDED_MODEL_PROTOCOL.md"
FEATURES = OUT / "mert_layers.npy"
FEATURE_META = OUT / "mert_layers.json"
LABELS = ["electronic", "pop", "ambient", "rock"]
ONTOLOGY = {
    "electronic": ["electronic", "electronica", "edm", "house", "deephouse", "techno", "trance",
                   "idm", "dubstep", "drumnbass", "breakbeat", "triphop", "downtempo", "electropop", "synthpop"],
    "pop": ["pop", "poprock", "popfolk", "instrumentalpop", "electropop", "synthpop"],
    "ambient": ["ambient", "darkambient", "atmospheric", "chillout", "newage"],
    "rock": ["rock", "alternativerock", "hardrock", "instrumentalrock", "postrock", "punkrock",
             "classicrock", "bluesrock", "rocknroll", "grunge", "poprock"],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def broad_targets(row):
    tags = set(row["tags"])
    return [int(bool(tags & set(ONTOLOGY[label]))) for label in LABELS]


def representation(values, name):
    if name == "mean_transformer":
        return values[:, 1:].mean(axis=1)
    if name.startswith("layer_") and len(name) == 8:
        layer = int(name[-2:])
        if 0 <= layer <= 12:
            return values[:, layer]
    raise ValueError(f"Unknown representation: {name}")


def freeze():
    if PLAN.exists():
        raise FileExistsError("Expanded model plan already frozen.")
    manifest = load_frozen()
    train = [row for row in manifest["tracks"] if row["phase_role"] == "train"]
    validation = [row for row in manifest["tracks"] if row["phase_role"] == "development_validation"]
    y = np.array([broad_targets(row) for row in train])
    groups = np.array([row["artist_id"] for row in train])
    folds = np.full(len(train), -1, dtype=int)
    splitter = GroupKFold(5, shuffle=True, random_state=20260830)
    for fold, (fit, held) in enumerate(splitter.split(groups, groups=groups)):
        if set(groups[fit]) & set(groups[held]):
            raise ValueError("Artist leakage in frozen folds.")
        if np.any((y[fit].sum(axis=0) == 0) | (y[held].sum(axis=0) == 0)):
            raise ValueError("A frozen fold lacks a positive label.")
        folds[held] = fold
    representations = [f"layer_{index:02d}" for index in range(13)] + ["mean_transformer"]
    configs = [{"C": c, "class_weight": weight} for c in (0.001, 0.01, 0.1)
               for weight in (None, "balanced")]
    plan = {
        "protocol_sha256": digest(PROTOCOL),
        "expanded_manifest_sha256": digest(MANIFEST),
        "model_source_sha256": digest(Path(__file__)),
        "layer_source_sha256": digest(ROOT / "mert_layer_features.py"),
        "extraction_source_sha256": digest(ROOT / "extract_expanded_mert.py"),
        "encoder_manifest_sha256": digest(ROOT / "models/mert-v0-public/SOURCES.json"),
        "labels": LABELS, "ontology": ONTOLOGY,
        "training_ids": [row["track_id"] for row in train],
        "validation_ids": [row["track_id"] for row in validation],
        "excluded_test_ids": [row["track_id"] for row in manifest["tracks"] if row["split"] == "test"],
        "fold_by_training_row": folds.tolist(), "seed": 20260830,
        "representations": representations, "configs": configs,
        "f1_grid": np.linspace(0.05, 0.95, 37).tolist(),
        "precision_grid": np.linspace(0.05, 0.95, 37).tolist(),
        "target_precision": 0.8, "min_recall": 0.3, "min_oof_predictions": 20,
    }
    if set(plan["training_ids"] + plan["validation_ids"]) & set(plan["excluded_test_ids"]):
        raise ValueError("Test ID entered the development plan.")
    save_json(PLAN, plan)
    print(f"Frozen {len(train)} train / {len(validation)} validation rows; "
          f"excluded {len(plan['excluded_test_ids'])} test rows.")


def load_development():
    plan = json.loads(PLAN.read_text())
    checks = ((PROTOCOL, "protocol_sha256"), (MANIFEST, "expanded_manifest_sha256"),
              (Path(__file__), "model_source_sha256"),
              (ROOT / "mert_layer_features.py", "layer_source_sha256"),
              (ROOT / "extract_expanded_mert.py", "extraction_source_sha256"),
              (ROOT / "models/mert-v0-public/SOURCES.json", "encoder_manifest_sha256"))
    for path, key in checks:
        if digest(path) != plan[key]:
            raise ValueError(f"Frozen model input changed: {Path(path).relative_to(ROOT)}")
    if plan["ontology"] != ONTOLOGY or plan["labels"] != LABELS:
        raise ValueError("Frozen broad-genre ontology changed.")
    manifest = load_frozen()
    by_id = {row["track_id"]: row for row in manifest["tracks"]}
    ids = plan["training_ids"] + plan["validation_ids"]
    rows = [by_id[track_id] for track_id in ids]
    if any(row["phase_role"] not in {"train", "development_validation"} for row in rows):
        raise ValueError("Non-development row entered expanded model.")
    if set(ids) & set(plan["excluded_test_ids"]):
        raise ValueError("Test ID entered expanded model.")
    audit = json.loads(AUDIT.read_text())
    if audit["manifest_sha256"] != digest(MANIFEST) or set(audit["tracks"]) != set(ids):
        raise ValueError("Expanded audio audit differs from model rows.")
    y = np.array([broad_targets(row) for row in rows])
    groups = np.array([row["artist_id"] for row in rows])
    return plan, rows, y, groups


def load_features(plan, rows):
    meta = json.loads(FEATURE_META.read_text())
    ids = [row["track_id"] for row in rows]
    if meta["model_plan_sha256"] != digest(PLAN) or meta["feature_sha256"] != digest(FEATURES):
        raise ValueError("Expanded MERT cache provenance differs from plan.")
    if meta["ids"] != ids or meta["test_tracks"] != 0:
        raise ValueError("Expanded MERT cache contains the wrong IDs.")
    values = np.load(FEATURES, mmap_mode="r", allow_pickle=False)
    if values.shape != (len(rows), 13, 768) or not np.isfinite(values).all():
        raise ValueError("Invalid expanded MERT feature matrix.")
    return values


def classifier(config):
    return make_pipeline(StandardScaler(), LogisticRegression(
        **config, max_iter=3000, random_state=2026))


def cross_validate(x, target, groups, folds, config):
    scores = np.full(len(target), np.nan)
    fold_ap = []
    for fold in np.unique(folds):
        fit, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if set(groups[fit]) & set(groups[held]):
            raise ValueError("Artist leakage during expanded CV.")
        model = classifier(config).fit(x[fit], target[fit])
        scores[held] = model.predict_proba(x[held])[:, 1]
        fold_ap.append(float(average_precision_score(target[held], scores[held])))
    if not np.isfinite(scores).all():
        raise ValueError("Incomplete expanded OOF scores.")
    return scores, fold_ap


def choose_f1(target, scores, grid):
    candidates = [(f1_score(target, scores >= threshold, zero_division=0),
                   -abs(threshold - 0.5), -threshold, threshold) for threshold in grid]
    return max(candidates)[-1]


def choose_precision(target, scores, plan):
    candidates = []
    positives = int(target.sum())
    for threshold in plan["precision_grid"]:
        predicted = scores >= threshold
        count = int(predicted.sum())
        tp = int((predicted & (target == 1)).sum())
        precision = tp / max(count, 1)
        recall = tp / max(positives, 1)
        if count >= plan["min_oof_predictions"] and precision >= plan["target_precision"] and recall >= plan["min_recall"]:
            candidates.append((recall, precision, -abs(threshold - 0.5), -threshold, threshold))
    return (max(candidates)[-1], True) if candidates else (1.01, False)


def evaluate(targets, scores, thresholds):
    predicted = scores >= thresholds
    tp = (predicted & (targets == 1)).sum(axis=0)
    fp = (predicted & (targets == 0)).sum(axis=0)
    fn = (~predicted & (targets == 1)).sum(axis=0)
    per_label = []
    for j, label in enumerate(LABELS):
        precision = float(tp[j] / max(tp[j] + fp[j], 1))
        recall = float(tp[j] / max(tp[j] + fn[j], 1))
        per_label.append({"label": label, "positives": int(targets[:, j].sum()),
                          "tp": int(tp[j]), "fp": int(fp[j]), "fn": int(fn[j]),
                          "predictions": int(tp[j] + fp[j]), "precision": precision,
                          "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-15),
                          "average_precision": float(average_precision_score(targets[:, j], scores[:, j]))})
    micro_precision = float(tp.sum() / max((tp + fp).sum(), 1))
    micro_recall = float(tp.sum() / max((tp + fn).sum(), 1))
    coverage = float(predicted.any(axis=1).mean())
    return {
        "micro_precision": micro_precision, "micro_recall": micro_recall,
        "micro_f1": 2 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-15),
        "macro_precision": float(np.mean([row["precision"] for row in per_label])),
        "macro_recall": float(np.mean([row["recall"] for row in per_label])),
        "macro_f1": float(np.mean([row["f1"] for row in per_label])),
        "macro_ap": float(np.mean([row["average_precision"] for row in per_label])),
        "track_output_coverage": coverage, "tracks_with_output": int(predicted.any(axis=1).sum()),
        "per_label": per_label,
        "provisional_goal_met": bool(micro_precision >= 0.8 and coverage >= 0.5 and all(
            row["precision"] >= 0.8 and row["recall"] >= 0.3 and row["predictions"] >= 10
            for row in per_label)),
    }


def run():
    plan, rows, y, groups = load_development()
    values = load_features(plan, rows)
    n = len(plan["training_ids"])
    folds = np.array(plan["fold_by_training_row"])
    if len(folds) != n:
        raise ValueError("Frozen fold vector length changed.")
    trials, selected, oof_columns, validation_columns, models = {}, [], [], [], []
    for j, label in enumerate(LABELS):
        label_trials, best = [], None
        for rep_name in plan["representations"]:
            x = representation(values[:n], rep_name)
            for config in plan["configs"]:
                scores, fold_ap = cross_validate(x, y[:n, j], groups[:n], folds, config)
                trial = {"representation": rep_name, "config": config,
                         "fold_ap": fold_ap, "mean_ap": float(np.mean(fold_ap))}
                label_trials.append(trial)
                if best is None or trial["mean_ap"] > best[0]["mean_ap"]:
                    best = (trial, scores)
        trial, oof_scores = best
        x_train = representation(values[:n], trial["representation"])
        x_validation = representation(values[n:], trial["representation"])
        model = classifier(trial["config"]).fit(x_train, y[:n, j])
        validation_scores = model.predict_proba(x_validation)[:, 1]
        trials[label] = label_trials
        selected.append(trial)
        oof_columns.append(oof_scores)
        validation_columns.append(validation_scores)
        models.append(model)
        print(f"Selected {label}: {trial['representation']} {trial['config']} "
              f"CV AP={trial['mean_ap']:.4f}", flush=True)
    oof = np.column_stack(oof_columns)
    validation_scores = np.column_stack(validation_columns)
    f1_thresholds = np.array([choose_f1(y[:n, j], oof[:, j], plan["f1_grid"])
                              for j in range(len(LABELS))])
    precise = [choose_precision(y[:n, j], oof[:, j], plan) for j in range(len(LABELS))]
    precision_thresholds = np.array([value for value, _ in precise])
    supported = [supported for _, supported in precise]
    policies = {"f1": f1_thresholds.tolist(), "precision_target": precision_thresholds.tolist()}
    report = {
        "plan_sha256": digest(PLAN), "feature_sha256": digest(FEATURES),
        "versions": {package: version(package) for package in ("numpy", "scikit-learn", "scipy", "joblib")},
        "target_definition": "frozen broad-genre ontology", "labels": LABELS,
        "train_count": n, "validation_count": len(rows) - n, "test_prediction_count": 0,
        "selected": selected, "trials": trials, "precision_supported": supported,
        "thresholds": policies,
        "oof": {"f1": evaluate(y[:n], oof, f1_thresholds),
                "precision_target": evaluate(y[:n], oof, precision_thresholds)},
        "validation": {"f1": evaluate(y[n:], validation_scores, f1_thresholds),
                       "precision_target": evaluate(y[n:], validation_scores, precision_thresholds)},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "development_scores.npz", ids=np.array([row["track_id"] for row in rows]),
                        y=y, oof_scores=oof, validation_scores=validation_scores)
    joblib.dump({"models": models, "selected": selected, "thresholds": policies,
                 "precision_supported": supported, "labels": LABELS, "ontology": ONTOLOGY,
                 "plan_sha256": digest(PLAN), "evaluation_scope": "development only"},
                OUT / "broad_genre_candidate.joblib")
    save_json(OUT / "development_metrics.json", report)
    load_development()
    print(json.dumps({"selected": selected, "precision_supported": supported,
                      "validation": report["validation"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    (freeze if args.command == "freeze" else run)()


if __name__ == "__main__":
    main()
