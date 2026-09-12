"""Analyze errors in the completed MAEST development experiment."""

from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from diagnose_expanded import FRACTIONS, top_k_diagnostic
from expanded_maest_hf import OUT, load_features, load_plan
from expanded_model import LABELS, classifier, load_development
from prepare_dataset import save_json


ROOT = Path(__file__).resolve().parent
REVIEW = ROOT / "data/maest_error_listening_review.csv"
NEIGHBOURS = {
    "electronic": {"dance", "experimental", "industrial"},
    "pop": {"chanson", "indie", "singersongwriter"},
    "ambient": {"easylistening", "experimental", "lounge", "soundscape"},
    "rock": {"alternative", "metal", "indie"},
}


def learning_curve(values, y, groups, folds, selected):
    result = []
    for j, label in enumerate(LABELS):
        config = selected[j]["config"]
        for fold in sorted(set(folds)):
            pool, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            artists = np.unique(groups[pool])
            artists = np.random.default_rng(20260830 + 100 * j + fold).permutation(artists)
            for fraction in FRACTIONS:
                chosen = artists[:max(1, round(len(artists) * fraction))]
                fit = pool[np.isin(groups[pool], chosen)]
                target = y[fit, j]
                if target.sum() == 0 or target.sum() == len(target):
                    raise ValueError("A diagnostic learning subset lacks both classes.")
                score = classifier(config).fit(values[fit], target).predict_proba(values[held])[:, 1]
                result.append({"label": label, "fold": int(fold), "fraction": fraction,
                               "fit_tracks": len(fit), "fit_artists": len(chosen),
                               "held_tracks": len(held), "held_average_precision": float(
                                   average_precision_score(y[held, j], score))})
    return result


def error_audit(rows, y, scores, thresholds):
    predicted, summaries, cases = scores >= thresholds, [], []
    for j, label in enumerate(LABELS):
        fp = np.flatnonzero(predicted[:, j] & (y[:, j] == 0))
        fn = np.flatnonzero(~predicted[:, j] & (y[:, j] == 1))
        fp = fp[np.argsort(-scores[fp, j], kind="stable")]
        fn = fn[np.argsort(scores[fn, j], kind="stable")]
        fp_tags = Counter(tag for index in fp for tag in rows[index]["tags"])
        fn_tags = Counter(tag for index in fn for tag in rows[index]["tags"])
        neighbour_count = sum(bool(set(rows[index]["tags"]) & NEIGHBOURS[label]) for index in fp)
        summaries.append({"label": label, "false_positives": len(fp), "false_negatives": len(fn),
                          "false_positives_with_neighbour_tag": neighbour_count,
                          "top_false_positive_source_tags": fp_tags.most_common(12),
                          "top_false_negative_source_tags": fn_tags.most_common(12)})
        for kind, indices in (("false_positive", fp), ("false_negative", fn)):
            for rank, index in enumerate(indices[:5], 1):
                cases.append({"label": label, "error_type": kind, "severity_rank": rank,
                              "track_id": rows[index]["track_id"], "artist": rows[index]["artist"],
                              "title": rows[index]["title"], "score": float(scores[index, j]),
                              "threshold": float(thresholds[j]), "source_tags": rows[index]["tags"],
                              "neighbour_tags": sorted(set(rows[index]["tags"]) & NEIGHBOURS[label]),
                              "audio_file": rows[index]["audio_file"], "track_url": rows[index]["track_url"]})
    return summaries, cases


def write_listening_review(cases, path=REVIEW):
    """Create a new sheet exclusively; never replace existing human work."""
    fields = ("label", "error_type", "severity_rank", "track_id", "artist", "title", "score",
              "threshold", "source_tags", "neighbour_tags", "audio_file", "auditor_hears_label",
              "auditor_reason")
    with Path(path).open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in cases:
            writer.writerow({**{key: row.get(key, "") for key in fields},
                             "source_tags": ";".join(row["source_tags"]),
                             "neighbour_tags": ";".join(row["neighbour_tags"])})


def main():
    # Fail before expensive diagnostics or any output writes. Exclusive creation
    # below also prevents a concurrent writer from being overwritten.
    if REVIEW.exists():
        raise FileExistsError(f"Listening sheet already exists; preserving it unchanged: {REVIEW}")
    plan = load_plan(); _, rows, y, groups = load_development(); values = load_features(plan, rows)
    metrics = json.loads((OUT / "development_metrics.json").read_text())
    with np.load(OUT / "development_scores.npz", allow_pickle=False) as data:
        if not np.array_equal(data["y"], y):
            raise ValueError("Saved labels differ from the frozen development rows.")
        validation_scores = data["validation_scores"]
    n = len(plan["training_ids"]); validation_rows, validation_y = rows[n:], y[n:]
    top_k = {label: top_k_diagnostic(validation_y[:, j], validation_scores[:, j])
             for j, label in enumerate(LABELS)}
    curve = learning_curve(values[:n], y[:n], groups[:n],
                           np.array(plan["fold_by_training_row"]), metrics["selected"])
    curve_summary = {label: [{"fraction": fraction,
                              "mean_fit_tracks": float(np.mean([r["fit_tracks"] for r in curve
                                  if r["label"] == label and r["fraction"] == fraction])),
                              "mean_held_ap": float(np.mean([r["held_average_precision"] for r in curve
                                  if r["label"] == label and r["fraction"] == fraction]))}
                             for fraction in FRACTIONS] for label in LABELS}
    thresholds = np.array(metrics["thresholds"]["f1"])
    audit, cases = error_audit(validation_rows, validation_y, validation_scores, thresholds)
    result = {"scope": "post-outcome exploratory MAEST development diagnosis",
              "ground_truth_warning": "Source tags and neighbours are listening prompts, not corrected labels.",
              "test_prediction_count": 0, "top_k_validation": top_k,
              "learning_curve": curve, "learning_curve_summary": curve_summary,
              "heuristic_neighbours": {k: sorted(v) for k, v in NEIGHBOURS.items()},
              "error_summary": audit, "listening_cases": cases}
    write_listening_review(cases)
    save_json(OUT / "post_outcome_diagnostic.json", result)
    print(json.dumps({"top_k_validation": top_k, "learning_curve_summary": curve_summary,
                      "error_summary": audit, "listening_cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
