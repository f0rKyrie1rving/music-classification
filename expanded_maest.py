"""Freeze and run the Discogs-MAEST broad-genre development comparison."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np

from expanded_model import (LABELS, ONTOLOGY, classifier, cross_validate,
                            evaluate, load_development, choose_f1, choose_precision)
from mert_features import sha256
from prepare_dataset import save_json
from prepare_expansion import MANIFEST, ROOT


BASE_PLAN = ROOT / "experiments/expanded_model_plan.json"
PLAN = ROOT / "experiments/maest_plan.json"
PROTOCOL = ROOT / "experiments/MAEST_PROTOCOL.md"
LOCK = ROOT / "requirements-maest-macos-arm64-lock.txt"
RECEIPT = ROOT / "models/discogs-maest-30s-pw-519l/SOURCES.json"
OUT = ROOT / "outputs/maest"
FEATURES = OUT / "features.npy"
FEATURE_META = OUT / "features.json"
REPRESENTATION = "maest_layer07_cls_dist_signal_mean"


def freeze():
    if PLAN.exists():
        raise FileExistsError("MAEST plan already frozen.")
    base = json.loads(BASE_PLAN.read_text())
    keys = ("labels", "ontology", "training_ids", "validation_ids", "excluded_test_ids",
            "fold_by_training_row", "seed", "configs", "f1_grid", "precision_grid",
            "target_precision", "min_recall", "min_oof_predictions")
    plan = {key: base[key] for key in keys}
    plan.update({
        "candidate": "UPF/MTG discogs-maest-30s-pw-519l-2",
        "representations": [REPRESENTATION],
        "protocol_sha256": sha256(PROTOCOL), "base_plan_sha256": sha256(BASE_PLAN),
        "expanded_manifest_sha256": sha256(MANIFEST), "runtime_lock_sha256": sha256(LOCK),
        "runner_source_sha256": sha256(Path(__file__)),
        "downloader_source_sha256": sha256(ROOT / "prepare_maest.py"),
        "extractor_source_sha256": sha256(ROOT / "maest_features.py"),
        "driver_source_sha256": sha256(ROOT / "extract_expanded_maest.py"),
        "encoder_manifest_sha256": sha256(RECEIPT),
    })
    if set(plan["training_ids"] + plan["validation_ids"]) & set(plan["excluded_test_ids"]):
        raise ValueError("Test ID entered MAEST development plan.")
    save_json(PLAN, plan)
    print(f"Frozen MAEST plan: {len(plan['training_ids'])} train, "
          f"{len(plan['validation_ids'])} validation, "
          f"{len(plan['excluded_test_ids'])} excluded test IDs.")


def load_plan():
    plan = json.loads(PLAN.read_text())
    checks = ((PROTOCOL, "protocol_sha256"), (BASE_PLAN, "base_plan_sha256"),
              (MANIFEST, "expanded_manifest_sha256"), (LOCK, "runtime_lock_sha256"),
              (Path(__file__), "runner_source_sha256"),
              (ROOT / "prepare_maest.py", "downloader_source_sha256"),
              (ROOT / "maest_features.py", "extractor_source_sha256"),
              (ROOT / "extract_expanded_maest.py", "driver_source_sha256"),
              (RECEIPT, "encoder_manifest_sha256"))
    for path, key in checks:
        if sha256(path) != plan[key]:
            raise ValueError(f"Frozen MAEST input changed: {Path(path).relative_to(ROOT)}")
    base = json.loads(BASE_PLAN.read_text())
    for key in ("labels", "ontology", "training_ids", "validation_ids", "excluded_test_ids",
                "fold_by_training_row", "configs", "f1_grid", "precision_grid",
                "target_precision", "min_recall", "min_oof_predictions"):
        if plan[key] != base[key]:
            raise ValueError(f"MAEST plan differs from the fixed v0 protocol: {key}")
    if plan["representations"] != [REPRESENTATION]:
        raise ValueError("MAEST representation search space changed.")
    return plan


def load_features(plan, rows):
    meta = json.loads(FEATURE_META.read_text())
    ids = [row["track_id"] for row in rows]
    values = np.load(FEATURES, mmap_mode="r", allow_pickle=False)
    if (meta["model_plan_sha256"] != sha256(PLAN)
            or meta["feature_sha256"] != sha256(FEATURES) or meta["ids"] != ids
            or meta["test_tracks"] != 0 or values.shape != (len(rows), 2304)
            or not np.isfinite(values).all()):
        raise ValueError("MAEST cache provenance or values are invalid.")
    return values


def run():
    plan = load_plan()
    base_plan, rows, y, groups = load_development()
    if ([row["track_id"] for row in rows] != plan["training_ids"] + plan["validation_ids"]
            or base_plan["ontology"] != ONTOLOGY):
        raise ValueError("MAEST rows or ontology differ from the fixed comparison.")
    values = load_features(plan, rows)
    n = len(plan["training_ids"])
    folds = np.array(plan["fold_by_training_row"])
    trials, selected, oof_columns, validation_columns, models = {}, [], [], [], [],
    for j, label in enumerate(LABELS):
        label_trials, best = [], None
        for config in plan["configs"]:
            scores, fold_ap = cross_validate(values[:n], y[:n, j], groups[:n], folds, config)
            trial = {"representation": REPRESENTATION, "config": config,
                     "fold_ap": fold_ap, "mean_ap": float(np.mean(fold_ap))}
            label_trials.append(trial)
            if best is None or trial["mean_ap"] > best[0]["mean_ap"]:
                best = (trial, scores)
        trial, oof_scores = best
        model = classifier(trial["config"]).fit(values[:n], y[:n, j])
        trials[label] = label_trials; selected.append(trial); oof_columns.append(oof_scores)
        validation_columns.append(model.predict_proba(values[n:])[:, 1]); models.append(model)
        print(f"MAEST selected {label}: {trial['config']} CV AP={trial['mean_ap']:.4f}", flush=True)
    oof, validation_scores = np.column_stack(oof_columns), np.column_stack(validation_columns)
    f1_thresholds = np.array([choose_f1(y[:n, j], oof[:, j], plan["f1_grid"])
                              for j in range(len(LABELS))])
    precise = [choose_precision(y[:n, j], oof[:, j], plan) for j in range(len(LABELS))]
    precision_thresholds = np.array([value for value, _ in precise])
    supported = [flag for _, flag in precise]
    policies = {"f1": f1_thresholds.tolist(), "precision_target": precision_thresholds.tolist()}
    report = {"candidate": plan["candidate"], "plan_sha256": sha256(PLAN),
              "feature_sha256": sha256(FEATURES),
              "versions": {package: version(package) for package in
                           ("numpy", "scikit-learn", "scipy", "joblib")},
              "target_definition": "same frozen broad-genre ontology as MERT comparisons",
              "labels": LABELS, "train_count": n, "validation_count": len(rows)-n,
              "test_prediction_count": 0, "selected": selected, "trials": trials,
              "precision_supported": supported, "thresholds": policies,
              "oof": {"f1": evaluate(y[:n], oof, f1_thresholds),
                      "precision_target": evaluate(y[:n], oof, precision_thresholds)},
              "validation": {"f1": evaluate(y[n:], validation_scores, f1_thresholds),
                             "precision_target": evaluate(y[n:], validation_scores, precision_thresholds)}}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "development_scores.npz",
                        ids=np.array([row["track_id"] for row in rows]), y=y,
                        oof_scores=oof, validation_scores=validation_scores)
    joblib.dump({"models": models, "selected": selected, "thresholds": policies,
                 "precision_supported": supported, "labels": LABELS, "ontology": ONTOLOGY,
                 "plan_sha256": sha256(PLAN), "evaluation_scope": "development only"},
                OUT / "broad_genre_candidate.joblib")
    save_json(OUT / "development_metrics.json", report)
    load_plan(); load_development()
    print(json.dumps({"selected": selected, "precision_supported": supported,
                      "validation": report["validation"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    (freeze if args.command == "freeze" else run)()
