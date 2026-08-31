"""Freeze and run the one-time final MAEST holdout evaluation."""

import argparse
import csv
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np

from expanded_maest_hf import load_features as load_development_features
from expanded_maest_hf import load_plan as load_maest_plan
from expanded_model import LABELS, ONTOLOGY, broad_targets, classifier, evaluate, load_development
from mert_features import sha256
from prepare_dataset import save_json
from prepare_expansion import MANIFEST, ROOT, load_frozen


PLAN = ROOT / "experiments/final_holdout_plan.json"
PROTOCOL = ROOT / "experiments/FINAL_HOLDOUT_PROTOCOL.md"
MAEST_PLAN = ROOT / "experiments/maest_hf_plan.json"
DEVELOPMENT_METRICS = ROOT / "outputs/maest_hf/development_metrics.json"
DEVELOPMENT_FEATURES = ROOT / "outputs/maest_hf/features.npy"
DEVELOPMENT_FEATURE_META = ROOT / "outputs/maest_hf/features.json"
DEVELOPMENT_LISTENING = ROOT / "data/maest_error_listening_review.csv"
HOLDOUT_AUDIT = ROOT / "data/final_holdout_audit.json"
HOLDOUT_FEATURES = ROOT / "outputs/final_maest/holdout_features.npy"
HOLDOUT_FEATURE_META = ROOT / "outputs/final_maest/holdout_features.json"
BLIND_REVIEW = ROOT / "data/final_blind_review.csv"
MODEL_RECEIPT = ROOT / "models/mtg-upf-maest-519l/SOURCES.json"
LOCK = ROOT / "requirements-mert-lock.txt"
OUT = ROOT / "outputs/final_maest"
RESULTS = OUT / "holdout_metrics.json"
SEED, BOOTSTRAPS = 20260830, 2000


def select_blind_queries(rows, seed=SEED):
    """Select five source-positive and five source-negative unique tracks per label."""
    rng = np.random.default_rng(seed); targets = np.array([broad_targets(row) for row in rows])
    used, queries = set(), []
    for label_index, label in enumerate(LABELS):
        for source_target in (1, 0):
            eligible = np.array([index for index in range(len(rows))
                                 if index not in used and targets[index, label_index] == source_target])
            if len(eligible) < 5:
                raise ValueError(f"Not enough unique blind-review rows for {label}={source_target}.")
            chosen = rng.choice(eligible, size=5, replace=False)
            for index in chosen:
                used.add(int(index)); row = rows[int(index)]
                queries.append({"query_id": len(queries) + 1, "label": label,
                                "track_id": row["track_id"], "source_target": source_target,
                                "artist": row["artist"], "title": row["title"],
                                "audio_file": row["audio_file"]})
    if len(queries) != 40 or len({item["track_id"] for item in queries}) != 40:
        raise ValueError("Blind-review selection is not 40 unique tracks.")
    return queries


def write_blind_review(queries):
    if BLIND_REVIEW.exists():
        raise FileExistsError("Final blind-review sheet already exists.")
    fields = ("query_id", "label", "track_id", "artist", "title", "audio_file",
              "auditor_hears_label", "auditor_reason")
    with BLIND_REVIEW.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        for query in queries:
            writer.writerow({key: query.get(key, "") for key in fields})


def freeze():
    if PLAN.exists():
        raise FileExistsError("Final holdout plan already frozen.")
    manifest = load_frozen(); maest_plan = load_maest_plan()
    development = json.loads(DEVELOPMENT_METRICS.read_text())
    if (development["test_prediction_count"] != 0 or development["labels"] != LABELS
            or development["plan_sha256"] != sha256(MAEST_PLAN)):
        raise ValueError("Completed MAEST development result is not the expected input.")
    rows = manifest["tracks"]
    fit_rows = [row for row in rows if row["phase_role"] in {"train", "development_validation"}]
    holdout_rows = [row for row in rows if row["phase_role"] == "new_holdout"]
    historical = [row for row in rows if row["phase_role"] == "historical_test_excluded"]
    if (len(fit_rows), len(holdout_rows), len(historical)) != (1206, 239, 90):
        raise ValueError("Final data-role counts changed.")
    fit_artists = {row["artist_id"] for row in fit_rows}
    holdout_artists = {row["artist_id"] for row in holdout_rows}
    if fit_artists & holdout_artists:
        raise ValueError("Final fit and holdout artists overlap.")
    selected = [{"label": label, "representation": item["representation"],
                 "config": item["config"]} for label, item in zip(LABELS, development["selected"])]
    if any(item["representation"] != "maest_block07_cls_dist_signal_mean" for item in selected):
        raise ValueError("Unexpected final MAEST representation.")
    thresholds = development["thresholds"]
    if (not np.allclose(thresholds["f1"], [0.4, 0.275, 0.375, 0.275])
            or not np.allclose(thresholds["precision_target"], [0.625, 1.01, 1.01, 0.525])
            or development["precision_supported"] != [True, False, False, True]):
        raise ValueError("Final MAEST thresholds differ from the documented choice.")
    queries = select_blind_queries(holdout_rows)
    plan = {"candidate": development["candidate"], "seed": SEED,
            "bootstrap_replicates": BOOTSTRAPS, "labels": LABELS, "ontology": ONTOLOGY,
            "primary_policy": "f1", "secondary_policy": "precision_target",
            "fit_rule": "refit fixed heads on all 1206 development tracks",
            "fit_ids": [row["track_id"] for row in fit_rows],
            "holdout_ids": [row["track_id"] for row in holdout_rows],
            "historical_test_ids": [row["track_id"] for row in historical],
            "fit_artist_count": len(fit_artists), "holdout_artist_count": len(holdout_artists),
            "selected": selected, "thresholds": thresholds,
            "precision_supported": development["precision_supported"],
            "blind_review_queries": queries,
            "protocol_sha256": sha256(PROTOCOL),
            "expanded_manifest_sha256": sha256(MANIFEST),
            "maest_plan_sha256": sha256(MAEST_PLAN),
            "development_metrics_sha256": sha256(DEVELOPMENT_METRICS),
            "development_features_sha256": sha256(DEVELOPMENT_FEATURES),
            "development_feature_meta_sha256": sha256(DEVELOPMENT_FEATURE_META),
            "development_listening_sha256": sha256(DEVELOPMENT_LISTENING),
            "runtime_lock_sha256": sha256(LOCK), "model_receipt_sha256": sha256(MODEL_RECEIPT),
            "runner_source_sha256": sha256(Path(__file__)),
            "preparer_source_sha256": sha256(ROOT / "prepare_holdout.py"),
            "holdout_extractor_source_sha256": sha256(ROOT / "extract_holdout_maest_hf.py"),
            "feature_source_sha256": sha256(ROOT / "maest_hf_features.py"),
            "repair_source_sha256": sha256(ROOT / "repair_short_audio.py")}
    save_json(PLAN, plan); write_blind_review(queries)
    print(f"Frozen final plan: {len(fit_rows)} fit, {len(holdout_rows)} holdout, "
          f"{len(historical)} historical excluded, {len(queries)} blind queries.")


def load_plan():
    plan = json.loads(PLAN.read_text())
    checks = ((PROTOCOL, "protocol_sha256"), (MANIFEST, "expanded_manifest_sha256"),
              (MAEST_PLAN, "maest_plan_sha256"), (DEVELOPMENT_METRICS, "development_metrics_sha256"),
              (DEVELOPMENT_FEATURES, "development_features_sha256"),
              (DEVELOPMENT_FEATURE_META, "development_feature_meta_sha256"),
              (DEVELOPMENT_LISTENING, "development_listening_sha256"),
              (LOCK, "runtime_lock_sha256"), (MODEL_RECEIPT, "model_receipt_sha256"),
              (Path(__file__), "runner_source_sha256"),
              (ROOT / "prepare_holdout.py", "preparer_source_sha256"),
              (ROOT / "extract_holdout_maest_hf.py", "holdout_extractor_source_sha256"),
              (ROOT / "maest_hf_features.py", "feature_source_sha256"),
              (ROOT / "repair_short_audio.py", "repair_source_sha256"))
    for path, key in checks:
        if sha256(path) != plan[key]:
            raise ValueError(f"Frozen final input changed: {Path(path).relative_to(ROOT)}")
    if plan["labels"] != LABELS or plan["ontology"] != ONTOLOGY:
        raise ValueError("Final labels or ontology changed.")
    manifest = load_frozen(); by_id = {row["track_id"]: row for row in manifest["tracks"]}
    fit_rows = [by_id[track_id] for track_id in plan["fit_ids"]]
    holdout_rows = [by_id[track_id] for track_id in plan["holdout_ids"]]
    if (any(row["phase_role"] not in {"train", "development_validation"} for row in fit_rows)
            or any(row["phase_role"] != "new_holdout" for row in holdout_rows)
            or set(plan["fit_ids"]) & set(plan["holdout_ids"])):
        raise ValueError("Final plan crossed a data boundary.")
    return plan, fit_rows, holdout_rows


def load_holdout_features(plan):
    meta = json.loads(HOLDOUT_FEATURE_META.read_text())
    values = np.load(HOLDOUT_FEATURES, mmap_mode="r", allow_pickle=False)
    if (meta["plan_sha256"] != sha256(PLAN) or meta["feature_sha256"] != sha256(HOLDOUT_FEATURES)
            or meta["ids"] != plan["holdout_ids"] or meta["historical_test_tracks"] != 0
            or values.shape != (239, 2304) or not np.isfinite(values).all()):
        raise ValueError("Final holdout MAEST feature cache is invalid.")
    return values


def bootstrap_intervals(targets, scores, thresholds, replicates, seed):
    rng = np.random.default_rng(seed); global_keys = (
        "micro_precision", "micro_recall", "micro_f1", "macro_precision",
        "macro_recall", "macro_f1", "macro_ap", "track_output_coverage")
    samples = {key: [] for key in global_keys}
    per_label = {label: {key: [] for key in ("precision", "recall", "f1", "average_precision")}
                 for label in LABELS}
    for _ in range(replicates):
        indices = rng.integers(0, len(targets), size=len(targets))
        result = evaluate(targets[indices], scores[indices], thresholds)
        for key in global_keys: samples[key].append(result[key])
        for row in result["per_label"]:
            for key in per_label[row["label"]]: per_label[row["label"]][key].append(row[key])
    interval = lambda values: [float(value) for value in np.percentile(values, [2.5, 97.5])]
    return {"method": "track bootstrap percentile interval", "confidence": 0.95,
            "replicates": replicates, "seed": seed,
            "overall": {key: interval(values) for key, values in samples.items()},
            "per_label": {label: {key: interval(values) for key, values in metrics.items()}
                          for label, metrics in per_label.items()}}


def save_predictions(rows, targets, scores, policies):
    fields = ["track_id", "artist", "title", "source_tags"]
    for label in LABELS:
        fields += [f"source_{label}", f"score_{label}", f"predicted_f1_{label}",
                   f"predicted_precision_target_{label}"]
    with (OUT / "holdout_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        for index, row in enumerate(rows):
            record = {"track_id": row["track_id"], "artist": row["artist"],
                      "title": row["title"], "source_tags": ";".join(row["tags"])}
            for j, label in enumerate(LABELS):
                record.update({f"source_{label}": int(targets[index, j]),
                               f"score_{label}": float(scores[index, j]),
                               f"predicted_f1_{label}": int(scores[index, j] >= policies["f1"][j]),
                               f"predicted_precision_target_{label}": int(
                                   scores[index, j] >= policies["precision_target"][j])})
            writer.writerow(record)


def run():
    if RESULTS.exists():
        existing = json.loads(RESULTS.read_text())
        if existing.get("plan_sha256") != sha256(PLAN):
            raise ValueError("Existing final result belongs to a different plan.")
        print("Final holdout evaluation already exists; refusing to rescore.")
        print(json.dumps(existing["holdout"], indent=2)); return
    plan, fit_rows, holdout_rows = load_plan(); maest_plan, development_rows, y_fit, _ = load_development()
    if ([row["track_id"] for row in development_rows] != plan["fit_ids"]
            or maest_plan["ontology"] != ONTOLOGY):
        raise ValueError("Final fitting rows differ from frozen development rows.")
    x_fit = load_development_features(load_maest_plan(), development_rows)
    x_holdout = load_holdout_features(plan)
    y_holdout = np.array([broad_targets(row) for row in holdout_rows])
    models, columns = [], []
    for j, item in enumerate(plan["selected"]):
        if item["label"] != LABELS[j]:
            raise ValueError("Final selected-head order changed.")
        model = classifier(item["config"]).fit(x_fit, y_fit[:, j])
        models.append(model); columns.append(model.predict_proba(x_holdout)[:, 1])
        print(f"Final head fitted and scored: {item['label']}", flush=True)
    scores = np.column_stack(columns); policies = {
        key: np.array(value) for key, value in plan["thresholds"].items()}
    holdout = {}; intervals = {}
    for offset, key in enumerate(("f1", "precision_target")):
        holdout[key] = evaluate(y_holdout, scores, policies[key])
        intervals[key] = bootstrap_intervals(
            y_holdout, scores, policies[key], plan["bootstrap_replicates"], plan["seed"] + offset)
    report = {"candidate": plan["candidate"], "plan_sha256": sha256(PLAN),
              "development_feature_sha256": sha256(DEVELOPMENT_FEATURES),
              "holdout_feature_sha256": sha256(HOLDOUT_FEATURES),
              "labels": LABELS, "fit_count": len(fit_rows), "holdout_count": len(holdout_rows),
              "historical_test_prediction_count": 0, "new_holdout_prediction_count": len(holdout_rows),
              "selected": plan["selected"], "thresholds": plan["thresholds"],
              "precision_supported": plan["precision_supported"],
              "versions": {name: version(name) for name in
                           ("numpy", "scikit-learn", "scipy", "joblib")},
              "holdout": holdout, "confidence_intervals": intervals}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "holdout_scores.npz", ids=np.array(plan["holdout_ids"]),
                        y=y_holdout, scores=scores)
    save_predictions(holdout_rows, y_holdout, scores, policies)
    joblib.dump({"models": models, "selected": plan["selected"],
                 "thresholds": plan["thresholds"], "precision_supported": plan["precision_supported"],
                 "labels": LABELS, "ontology": ONTOLOGY, "plan_sha256": sha256(PLAN),
                 "evaluation_scope": "final model fitted on 1206 development tracks"},
                OUT / "final_broad_genre_model.joblib")
    save_json(RESULTS, report); load_plan(); load_holdout_features(plan)
    print(json.dumps(holdout, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args(); (freeze if args.command == "freeze" else run)()
