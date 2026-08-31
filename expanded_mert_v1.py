"""Freeze and run the MERT-v1 broad-genre development comparison."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np

from expanded_model import (LABELS, ONTOLOGY, classifier, cross_validate,
                            evaluate, load_development, representation,
                            choose_f1, choose_precision)
from mert_features import sha256
from prepare_dataset import save_json
from prepare_expansion import MANIFEST, ROOT


BASE_PLAN = ROOT / "experiments/expanded_model_plan.json"
FAILED_PLAN = ROOT / "experiments/mert_v1_plan.json"
IMPLEMENTATION_NOTE = ROOT / "experiments/MERT_V1_IMPLEMENTATION_NOTE.md"
PLAN = ROOT / "experiments/mert_v1_plan_v2.json"
PROTOCOL = ROOT / "experiments/MERT_V1_PROTOCOL.md"
OUT = ROOT / "outputs/mert_v1"
FEATURES = OUT / "mert_layers.npy"
FEATURE_META = OUT / "mert_layers.json"


def freeze():
    if PLAN.exists():
        raise FileExistsError("MERT-v1 plan already frozen.")
    base = json.loads(BASE_PLAN.read_text())
    plan = {key: base[key] for key in ("labels", "ontology", "training_ids", "validation_ids",
            "excluded_test_ids", "fold_by_training_row", "seed", "representations", "configs",
            "f1_grid", "precision_grid", "target_precision", "min_recall", "min_oof_predictions")}
    plan.update({
        "candidate": "m-a-p/MERT-v1-95M",
        "protocol_sha256": sha256(PROTOCOL), "base_plan_sha256": sha256(BASE_PLAN),
        "failed_plan_sha256": sha256(FAILED_PLAN),
        "implementation_note_sha256": sha256(IMPLEMENTATION_NOTE),
        "expanded_manifest_sha256": sha256(MANIFEST), "runner_source_sha256": sha256(Path(__file__)),
        "extractor_source_sha256": sha256(ROOT / "mert_v1_features.py"),
        "driver_source_sha256": sha256(ROOT / "extract_expanded_mert_v1.py"),
        "encoder_manifest_sha256": sha256(ROOT / "models/mert-v1-95m/SOURCES.json"),
    })
    if set(plan["training_ids"] + plan["validation_ids"]) & set(plan["excluded_test_ids"]):
        raise ValueError("Test ID entered MERT-v1 development plan.")
    save_json(PLAN, plan)
    print(f"Frozen MERT-v1 plan: {len(plan['training_ids'])} train, "
          f"{len(plan['validation_ids'])} validation, {len(plan['excluded_test_ids'])} excluded test IDs.")


def load_plan():
    plan = json.loads(PLAN.read_text())
    checks = ((PROTOCOL, "protocol_sha256"), (BASE_PLAN, "base_plan_sha256"),
              (FAILED_PLAN, "failed_plan_sha256"),
              (IMPLEMENTATION_NOTE, "implementation_note_sha256"),
              (MANIFEST, "expanded_manifest_sha256"), (Path(__file__), "runner_source_sha256"),
              (ROOT / "mert_v1_features.py", "extractor_source_sha256"),
              (ROOT / "extract_expanded_mert_v1.py", "driver_source_sha256"),
              (ROOT / "models/mert-v1-95m/SOURCES.json", "encoder_manifest_sha256"))
    for path, key in checks:
        if sha256(path) != plan[key]:
            raise ValueError(f"Frozen MERT-v1 input changed: {Path(path).relative_to(ROOT)}")
    base = json.loads(BASE_PLAN.read_text())
    for key in ("labels", "ontology", "training_ids", "validation_ids", "excluded_test_ids",
                "fold_by_training_row", "representations", "configs", "f1_grid", "precision_grid"):
        if plan[key] != base[key]:
            raise ValueError(f"MERT-v1 plan differs from v0 protocol: {key}")
    return plan


def load_features(plan, rows):
    meta = json.loads(FEATURE_META.read_text())
    ids = [row["track_id"] for row in rows]
    values = np.load(FEATURES, mmap_mode="r", allow_pickle=False)
    if (meta["model_plan_sha256"] != sha256(PLAN) or meta["feature_sha256"] != sha256(FEATURES)
            or meta["ids"] != ids or meta["test_tracks"] != 0
            or values.shape != (len(rows), 13, 768) or not np.isfinite(values).all()):
        raise ValueError("MERT-v1 cache provenance or values are invalid.")
    return values


def run():
    plan = load_plan()
    base_plan, rows, y, groups = load_development()
    if ([row["track_id"] for row in rows] != plan["training_ids"] + plan["validation_ids"]
            or base_plan["ontology"] != ONTOLOGY):
        raise ValueError("MERT-v1 rows or ontology differ from v0 comparison.")
    values = load_features(plan, rows)
    n = len(plan["training_ids"])
    folds = np.array(plan["fold_by_training_row"])
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
        trials[label] = label_trials
        selected.append(trial); oof_columns.append(oof_scores)
        validation_columns.append(model.predict_proba(x_validation)[:, 1]); models.append(model)
        print(f"MERT-v1 selected {label}: {trial['representation']} {trial['config']} "
              f"CV AP={trial['mean_ap']:.4f}", flush=True)
    oof = np.column_stack(oof_columns)
    validation_scores = np.column_stack(validation_columns)
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
              "target_definition": "same frozen broad-genre ontology as MERT-v0 comparison",
              "labels": LABELS, "train_count": n, "validation_count": len(rows)-n,
              "test_prediction_count": 0, "selected": selected, "trials": trials,
              "precision_supported": supported, "thresholds": policies,
              "oof": {"f1": evaluate(y[:n], oof, f1_thresholds),
                      "precision_target": evaluate(y[:n], oof, precision_thresholds)},
              "validation": {"f1": evaluate(y[n:], validation_scores, f1_thresholds),
                             "precision_target": evaluate(y[n:], validation_scores, precision_thresholds)}}
    np.savez_compressed(OUT / "development_scores.npz", ids=np.array([row["track_id"] for row in rows]),
                        y=y, oof_scores=oof, validation_scores=validation_scores)
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
