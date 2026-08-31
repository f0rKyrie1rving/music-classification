"""Post-outcome diagnosis for the expanded development experiment.

This analysis is explicitly exploratory.  It does not change a label, select a
released model, inspect test audio, or create a new performance claim.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from expanded_model import (FEATURES, LABELS, OUT, classifier, load_development,
                            load_features, representation)
from prepare_dataset import save_json


ROOT = Path(__file__).resolve().parent
FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def top_k_diagnostic(target, scores):
    order = np.argsort(-scores, kind="stable")
    rows = []
    for count in (5, 10, 20, 30, 40, 50):
        selected = order[:count]
        tp = int(target[selected].sum())
        rows.append({"count": count, "tp": tp, "precision": tp / count,
                     "recall": tp / int(target.sum()),
                     "minimum_score": float(scores[selected[-1]])})
    feasible = []
    for count in range(5, len(order) + 1):
        selected = order[:count]
        tp = int(target[selected].sum())
        precision = tp / count
        recall = tp / int(target.sum())
        if precision >= 0.8:
            feasible.append((recall, precision, -count, count,
                             float(scores[selected[-1]])))
    best = max(feasible) if feasible else None
    return {"top_k": rows, "best_at_least_80_precision": None if best is None else {
        "count": best[3], "precision": best[1], "recall": best[0], "minimum_score": best[4]}}


def learning_curve(values, y, groups, folds, selected):
    results = []
    for label_index, label in enumerate(LABELS):
        choice = selected[label_index]
        x = representation(values, choice["representation"])
        for fold in sorted(set(folds)):
            pool = np.flatnonzero(folds != fold)
            held = np.flatnonzero(folds == fold)
            artists = np.unique(groups[pool])
            artists = np.random.default_rng(20260830 + 100 * label_index + fold).permutation(artists)
            for fraction in FRACTIONS:
                chosen = artists[:max(1, round(len(artists) * fraction))]
                fit = pool[np.isin(groups[pool], chosen)]
                target = y[fit, label_index]
                if target.sum() == 0 or target.sum() == len(target):
                    raise ValueError("A diagnostic learning subset lacks both classes.")
                model = classifier(choice["config"]).fit(x[fit], target)
                score = model.predict_proba(x[held])[:, 1]
                results.append({"label": label, "fold": int(fold), "fraction": fraction,
                                "fit_tracks": len(fit), "fit_artists": len(chosen),
                                "held_tracks": len(held),
                                "held_average_precision": float(average_precision_score(
                                    y[held, label_index], score))})
    return results


def main():
    plan, rows, y, groups = load_development()
    values = load_features(plan, rows)
    metrics = json.loads((OUT / "development_metrics.json").read_text())
    with np.load(OUT / "development_scores.npz", allow_pickle=False) as scores:
        if not np.array_equal(scores["y"], y):
            raise ValueError("Diagnostic labels differ from the saved model run.")
        validation = scores["validation_scores"]
    n = len(plan["training_ids"])
    top_k = {label: top_k_diagnostic(y[n:, j], validation[:, j])
             for j, label in enumerate(LABELS)}
    curve = learning_curve(values[:n], y[:n], groups[:n],
                           np.array(plan["fold_by_training_row"]), metrics["selected"])
    summary = {}
    for label in LABELS:
        summary[label] = []
        for fraction in FRACTIONS:
            items = [row for row in curve if row["label"] == label and row["fraction"] == fraction]
            summary[label].append({"fraction": fraction,
                                   "mean_fit_tracks": float(np.mean([row["fit_tracks"] for row in items])),
                                   "mean_held_ap": float(np.mean([row["held_average_precision"] for row in items])),
                                   "fold_ap": [row["held_average_precision"] for row in items]})
    result = {"scope": "post-outcome exploratory development diagnosis",
              "test_prediction_count": 0, "fractions": list(FRACTIONS),
              "top_k_validation": top_k, "learning_curve": curve,
              "learning_curve_summary": summary}
    save_json(OUT / "post_expansion_diagnostic.json", result)
    print(json.dumps({"top_k_validation": top_k, "learning_curve_summary": summary}, indent=2))


if __name__ == "__main__":
    main()
