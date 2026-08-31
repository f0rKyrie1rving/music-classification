"""Bounded, artist-grouped development experiment; never evaluate old test rows."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from compare_features import check_protected, development_rows
from train import choose_thresholds, evaluate

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/improvement"
PLAN = ROOT / "experiments/improvement_plan.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_plan():
    if PLAN.exists():
        raise ValueError("Plan already frozen; do not overwrite it.")
    manifest = json.loads((ROOT / "data/dataset_manifest.json").read_text())
    rows = development_rows(manifest)
    training = [r for r in rows if r["split"] == "train"]
    groups = np.array([r["artist_id"] for r in training])
    folds = np.full(len(training), -1, dtype=int)
    for k, (fit, held) in enumerate(GroupKFold(3, shuffle=True, random_state=20260828).split(groups, groups=groups)):
        assert not set(groups[fit]) & set(groups[held])
        folds[held] = k
    plan = dict(protocol_sha256=digest(ROOT / "experiments/IMPROVEMENT_PROTOCOL.md"),
                source_sha256=digest(Path(__file__)),
                baseline_plan_sha256=digest(ROOT / "experiments/feature_comparison_plan.json"),
                labels=manifest["labels"], training_ids=[r["track_id"] for r in training],
                validation_ids=[r["track_id"] for r in rows if r["split"] == "validation"],
                fold_by_training_row=folds.tolist(), seed=20260828,
                grid=[dict(C=c, class_weight=w) for c in (.01, .1, 1.) for w in (None, "balanced")],
                learning_fractions=[.25, .5, 1.], target_precision=.8, min_recall=.3,
                min_oof_predictions=10, precision_grid=np.linspace(.05, .95, 37).tolist())
    PLAN.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Frozen {len(training)} training and {len(rows)-len(training)} validation IDs; no test rows.")


def load_development():
    plan = json.loads(PLAN.read_text())
    for path, key in [("experiments/IMPROVEMENT_PROTOCOL.md", "protocol_sha256"),
                      ("improve_model.py", "source_sha256"),
                      ("experiments/feature_comparison_plan.json", "baseline_plan_sha256")]:
        if digest(ROOT / path) != plan[key]:
            raise ValueError(f"Frozen definition changed: {path}")
    check_protected(json.loads((ROOT / "experiments/feature_comparison_plan.json").read_text()))
    manifest = json.loads((ROOT / "data/dataset_manifest.json").read_text())
    rows = development_rows(manifest)
    ids = plan["training_ids"] + plan["validation_ids"]
    by_id = {r["track_id"]: r for r in rows}
    if set(by_id) != set(ids) or manifest["labels"] != plan["labels"]:
        raise ValueError("Development manifest differs from the plan.")
    rows = [by_id[i] for i in ids]
    y = np.array([r["targets"] for r in rows])
    for row in rows:
        if row["targets"] != [int(t in row["tags"]) for t in plan["labels"]]:
            raise ValueError("Source targets changed.")
        if digest(ROOT / row["audio_file"]) != row["wav_sha256"]:
            raise ValueError("Development audio changed.")
    groups = np.array([r["artist_id"] for r in rows])
    return plan, rows, y, groups


def classifier(config):
    return make_pipeline(StandardScaler(), OneVsRestClassifier(LogisticRegression(
        **config, max_iter=2000, random_state=2026)))


def precision_thresholds(y, scores, plan):
    thresholds, supported = [], []
    for j in range(y.shape[1]):
        feasible = []
        for t in plan["precision_grid"]:
            positive = scores[:, j] >= t
            tp = int((positive & (y[:, j] == 1)).sum())
            n = int(positive.sum())
            p, r = tp / max(n, 1), tp / max(int(y[:, j].sum()), 1)
            if n >= plan["min_oof_predictions"] and p >= plan["target_precision"] and r >= plan["min_recall"]:
                feasible.append((r, p, -abs(t - .5), t))
        thresholds.append(max(feasible)[-1] if feasible else 1.01)
        supported.append(bool(feasible))
    return np.array(thresholds), supported


def operating_metrics(y, scores, thresholds, labels):
    report = evaluate(y, scores, thresholds, labels)
    predicted = scores >= thresholds
    tp = (predicted & (y == 1)).sum(axis=0)
    fp = (predicted & (y == 0)).sum(axis=0)
    fn = (~predicted & (y == 1)).sum(axis=0)
    for j, tag in enumerate(report["per_label"]):
        tag.update(tp=int(tp[j]), fp=int(fp[j]), fn=int(fn[j]), predictions=int(tp[j] + fp[j]),
                   precision_defined=bool(tp[j] + fp[j]))
    report.update(micro_precision=float(tp.sum() / max((tp + fp).sum(), 1)),
                  micro_recall=float(tp.sum() / max((tp + fn).sum(), 1)),
                  macro_precision=float(np.mean([r["precision"] for r in report["per_label"]])),
                  macro_recall=float(np.mean([r["recall"] for r in report["per_label"]])),
                  track_output_coverage=float(predicted.any(axis=1).mean()),
                  tracks_with_output=int(predicted.any(axis=1).sum()))
    report["provisional_goal_met"] = bool(report["micro_precision"] >= .8 and
        report["track_output_coverage"] >= .5 and all(
            r["precision"] >= .8 and r["recall"] >= .3 and r["predictions"] >= 5 for r in report["per_label"]))
    return report


def cross_validate(x, y, groups, folds, config):
    scores = np.full(y.shape, np.nan)
    results = []
    for k in np.unique(folds):
        fit, held = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        if set(groups[fit]) & set(groups[held]):
            raise ValueError("Artist leakage within a fold.")
        if any(np.any((y[idx].sum(axis=0) == 0) | (y[idx].sum(axis=0) == len(idx))) for idx in (fit, held)):
            raise ValueError("Fold lacks positive or negative examples.")
        model = classifier(config).fit(x[fit], y[fit])
        np.testing.assert_allclose(model[0].mean_, x[fit].mean(axis=0), rtol=0, atol=1e-12)
        scores[held] = model.predict_proba(x[held])
        results.append(float(average_precision_score(y[held], scores[held], average="macro")))
    if not np.isfinite(scores).all():
        raise ValueError("Incomplete OOF predictions.")
    return scores, results


def learning_curve(x, y, groups, folds, plan):
    results = []
    for k in np.unique(folds):
        pool, held = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        artists = np.random.default_rng(plan["seed"] + int(k)).permutation(np.unique(groups[pool]))
        for fraction in plan["learning_fractions"]:
            chosen = artists[:max(1, round(len(artists) * fraction))]
            fit = pool[np.isin(groups[pool], chosen)]
            if np.any((y[fit].sum(axis=0) == 0) | (y[fit].sum(axis=0) == len(fit))):
                raise ValueError("Learning subset lacks a class; report instead of resampling.")
            model = classifier(dict(C=1., class_weight="balanced")).fit(x[fit], y[fit])
            results.append(dict(fold=int(k), fraction=fraction, fit_tracks=len(fit), fit_artists=len(chosen),
                held_tracks=len(held), train_macro_ap=float(average_precision_score(y[fit], model.predict_proba(x[fit]), average="macro")),
                held_macro_ap=float(average_precision_score(y[held], model.predict_proba(x[held]), average="macro"))))
    return results


def run(name, cache_path=None):
    plan, rows, y, groups = load_development()
    ids = np.array([r["track_id"] for r in rows])
    cache_path = Path(cache_path) if cache_path else ROOT / "outputs/dataset_features.npz"
    with np.load(cache_path, allow_pickle=False) as cached:
        if len(set(cached["ids"])) != len(cached["ids"]):
            raise ValueError("Duplicate cached IDs.")
        if name == "mert" and set(cached["ids"]) != set(ids):
            raise ValueError("MERT cache must contain development tracks only.")
        positions = {v: i for i, v in enumerate(cached["ids"])}
        x = cached["x"][[positions[i] for i in ids]]
        if name == "mert":
            source_path = ROOT / "models/mert-v0-public/SOURCES.json"
            if str(cached["encoder_manifest_sha256"]) != digest(source_path):
                raise ValueError("Encoder provenance changed.")
            if str(cached["extractor_source_sha256"]) != digest(ROOT / "mert_features.py"):
                raise ValueError("Encoder extraction code changed.")
            if not np.array_equal(cached["audio_hashes"][[positions[i] for i in ids]],
                                  np.array([r["wav_sha256"] for r in rows])):
                raise ValueError("Encoder cache audio differs from development audio.")
            sources = json.loads(source_path.read_text())
            for file, sha256 in sources["files_sha256"].items():
                if digest(source_path.parent / file) != sha256:
                    raise ValueError(f"Encoder file changed: {file}")
    expected = 26 if name == "mfcc" else 768
    if x.shape != (len(rows), expected) or not np.isfinite(x).all():
        raise ValueError("Invalid feature matrix.")
    n = len(plan["training_ids"])
    folds = np.array(plan["fold_by_training_row"])
    OUT.mkdir(parents=True, exist_ok=True)
    trials, oof = [], []
    for config in plan["grid"]:
        scores, ap = cross_validate(x[:n], y[:n], groups[:n], folds, config)
        trials.append(dict(config=config, fold_macro_ap=ap, mean_macro_ap=float(np.mean(ap))))
        oof.append(scores)
    best = max(range(len(trials)), key=lambda i: trials[i]["mean_macro_ap"])
    model = classifier(trials[best]["config"]).fit(x[:n], y[:n])
    validation_scores = model.predict_proba(x[n:])
    f1_thresholds = choose_thresholds(y[:n], oof[best])
    precise, supported = precision_thresholds(y[:n], oof[best], plan)
    policies = {"f1": f1_thresholds, "precision_target": precise}
    report = dict(representation=name, plan_sha256=digest(PLAN), cache_sha256=digest(cache_path),
        versions={p: version(p) for p in ("numpy", "scikit-learn", "scipy", "joblib")},
        selection="mean artist-grouped CV Macro AP on 300 training tracks", trials=trials,
        selected_config=trials[best]["config"], train_count=n, validation_count=len(rows)-n,
        test_prediction_count=0, labels=plan["labels"], precision_supported=supported,
        thresholds={k: v.tolist() for k, v in policies.items()},
        validation={k: operating_metrics(y[n:], validation_scores, t, plan["labels"]) for k, t in policies.items()},
        oof={k: operating_metrics(y[:n], oof[best], t, plan["labels"]) for k, t in policies.items()})
    if name == "mfcc":
        report["learning_curve"] = learning_curve(x[:n], y[:n], groups[:n], folds, plan)
    np.savez_compressed(OUT / f"{name}_scores.npz", ids=ids, y=y, oof_scores=oof[best], validation_scores=validation_scores)
    joblib.dump(dict(model=model, policies=policies, labels=plan["labels"], plan_sha256=digest(PLAN),
                     evaluation_scope="development only"), OUT / f"{name}_head.joblib")
    load_development()  # Recheck protected artifacts after all fits.
    (OUT / f"{name}_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("selected_config", "precision_supported", "validation")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "mfcc", "mert"])
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze_plan()
    else:
        run(args.action, args.cache)
