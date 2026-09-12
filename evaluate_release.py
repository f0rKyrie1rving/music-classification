"""Recompute published results and post-hoc checks without audio or local caches."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from report_blind_review import summarize

ROOT = Path(__file__).resolve().parent
LABELS = ("electronic", "pop", "ambient", "rock")
GLOBAL_KEYS = ("micro_precision", "micro_recall", "micro_f1", "macro_precision",
               "macro_recall", "macro_f1", "macro_ap", "track_output_coverage")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_baseline(root=ROOT):
    baseline = read_json(root / "experiments/review_baseline.json")
    for name, expected in baseline["sha256"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Historical protected file changed: {name}")
    return baseline


def load_release(root=ROOT):
    plan = read_json(root / "experiments/final_holdout_plan.json")
    manifest = read_json(root / "data/expanded_manifest.json")
    release = read_json(root / "data/evaluation/holdout.json")
    if release["plan_sha256"] != sha256(root / "experiments/final_holdout_plan.json"):
        raise ValueError("Published scores belong to a different final plan.")
    rows = release["rows"]
    if release["labels"] != list(LABELS) or [r["track_id"] for r in rows] != plan["holdout_ids"]:
        raise ValueError("Published holdout IDs or label order changed.")
    by_id = {r["track_id"]: r for r in manifest["tracks"]}
    targets = np.array([r["targets"] for r in rows])
    scores = np.array([r["scores"] for r in rows], dtype=np.float64)
    if (targets.shape != (239, 4) or scores.shape != targets.shape
            or not np.isin(targets, (0, 1)).all() or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1))):
        raise ValueError("Invalid published targets or scores.")
    for i, row in enumerate(rows):
        source = by_id[row["track_id"]]
        expected = [int(bool(set(source["tags"]) & set(plan["ontology"][label]))) for label in LABELS]
        if (source["phase_role"] != "new_holdout" or row["artist_id"] != source["artist_id"]
                or row["targets"] != expected):
            raise ValueError(f"Published row differs from frozen source: {row['track_id']}")
        for policy, thresholds in plan["thresholds"].items():
            if row["decisions"][policy] != (scores[i] >= np.array(thresholds)).astype(int).tolist():
                raise ValueError("Published decision differs from its frozen threshold.")
    fit_artists = {by_id[i]["artist_id"] for i in plan["fit_ids"]}
    if fit_artists & {r["artist_id"] for r in rows}:
        raise ValueError("Fit and holdout artists overlap.")
    return plan, manifest, rows, targets, scores


def evaluate(targets, scores, thresholds):
    """Independent calculation of the archived metric definitions."""
    predicted = scores >= np.array(thresholds)
    tp = (predicted & (targets == 1)).sum(axis=0)
    fp = (predicted & (targets == 0)).sum(axis=0)
    fn = (~predicted & (targets == 1)).sum(axis=0)
    per_label = []
    for j, label in enumerate(LABELS):
        precision = float(tp[j] / max(tp[j] + fp[j], 1))
        recall = float(tp[j] / max(tp[j] + fn[j], 1))
        ap = float(average_precision_score(targets[:, j], scores[:, j])) if targets[:, j].any() else 0.0
        per_label.append(dict(label=label, positives=int(targets[:, j].sum()), tp=int(tp[j]),
                              fp=int(fp[j]), fn=int(fn[j]), predictions=int(tp[j] + fp[j]),
                              precision=precision, recall=recall,
                              f1=2 * precision * recall / max(precision + recall, 1e-15),
                              average_precision=ap))
    precision = float(tp.sum() / max((tp + fp).sum(), 1))
    recall = float(tp.sum() / max((tp + fn).sum(), 1))
    coverage = float(predicted.any(axis=1).mean())
    return dict(micro_precision=precision, micro_recall=recall,
                micro_f1=2 * precision * recall / max(precision + recall, 1e-15),
                macro_precision=float(np.mean([r["precision"] for r in per_label])),
                macro_recall=float(np.mean([r["recall"] for r in per_label])),
                macro_f1=float(np.mean([r["f1"] for r in per_label])),
                macro_ap=float(np.mean([r["average_precision"] for r in per_label])),
                track_output_coverage=coverage, tracks_with_output=int(predicted.any(axis=1).sum()),
                per_label=per_label, provisional_goal_met=bool(precision >= .8 and coverage >= .5 and all(
                    r["precision"] >= .8 and r["recall"] >= .3 and r["predictions"] >= 10 for r in per_label)))


def bootstrap_indices(rng, count, groups=None):
    if groups is None:
        return rng.integers(0, count, size=count)
    artists = np.unique(groups)
    return np.concatenate([np.flatnonzero(groups == artist)
                           for artist in rng.choice(artists, size=len(artists), replace=True)])


def bootstrap(targets, scores, thresholds, replicates, seed, groups=None):
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in GLOBAL_KEYS}
    per_label = {label: {k: [] for k in ("precision", "recall", "f1", "average_precision")}
                 for label in LABELS}
    empty = {label: {"no_positives": 0, "no_predictions": 0} for label in LABELS}
    for _ in range(replicates):
        index = bootstrap_indices(rng, len(targets), groups)
        result = evaluate(targets[index], scores[index], thresholds)
        for key in samples:
            samples[key].append(result[key])
        for row in result["per_label"]:
            for key in per_label[row["label"]]:
                per_label[row["label"]][key].append(row[key])
            empty[row["label"]]["no_positives"] += int(row["positives"] == 0)
            empty[row["label"]]["no_predictions"] += int(row["predictions"] == 0)
    interval = lambda values: np.percentile(values, [2.5, 97.5]).tolist()
    return dict(method="track bootstrap percentile interval" if groups is None else
                "artist-cluster bootstrap percentile interval", confidence=.95,
                replicates=replicates, seed=seed,
                overall={k: interval(v) for k, v in samples.items()},
                per_label={label: {k: interval(v) for k, v in metrics.items()}
                           for label, metrics in per_label.items()}, empty_replicates=empty)


def assert_matches(actual, expected, location="result"):
    """Compare all archived fields; supplementary fields in actual are allowed."""
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert_matches(actual[key], value, f"{location}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"Length mismatch: {location}")
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            assert_matches(a, b, f"{location}[{i}]")
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not np.isclose(actual, expected, atol=1e-12, rtol=1e-10):
            raise ValueError(f"Numeric mismatch: {location}: {actual} != {expected}")
    elif actual != expected:
        raise ValueError(f"Mismatch: {location}: {actual} != {expected}")


def blind_summary(plan, rows, root=ROOT):
    with (root / "data/final_blind_review.csv").open(encoding="utf-8", newline="") as file:
        reviews = list(csv.DictReader(file))
    by_query = {int(r["query_id"]): r for r in reviews}
    by_id = {r["track_id"]: r for r in rows}
    if len(reviews) != 40 or len(by_query) != 40:
        raise ValueError("Expected 40 unique completed listening answers.")
    comparison = []
    for query in plan["blind_review_queries"]:
        review = by_query[query["query_id"]]
        if (review["track_id"], review["label"]) != (query["track_id"], query["label"]):
            raise ValueError("Listening query changed.")
        human = {"yes": 1, "no": 0, "uncertain": None}[review["auditor_hears_label"]]
        j = LABELS.index(query["label"])
        if query["source_target"] != by_id[query["track_id"]]["targets"][j]:
            raise ValueError("Listening source target changed.")
        comparison.append(dict(label=query["label"], human_decision=review["auditor_hears_label"],
                               human_target=human, source_target=query["source_target"],
                               model_prediction=by_id[query["track_id"]]["decisions"]["f1"][j]))
    return dict(overall=summarize(comparison), by_label={label: summarize(
        [r for r in comparison if r["label"] == label]) for label in LABELS})


def analyze(root=ROOT):
    baseline = check_baseline(root)
    plan, manifest, rows, targets, scores = load_release(root)
    original = read_json(root / "data/evaluation/original_metrics.json")
    archived_blind = read_json(root / "data/evaluation/original_blind_summary.json")
    if original["plan_sha256"] != sha256(root / "experiments/final_holdout_plan.json"):
        raise ValueError("Archived metrics belong to a different plan.")
    groups = np.array([r["artist_id"] for r in rows])
    policies, track_intervals, cluster_intervals = {}, {}, {}
    for offset, policy in enumerate(("f1", "precision_target")):
        thresholds = plan["thresholds"][policy]
        policies[policy] = evaluate(targets, scores, thresholds)
        assert_matches(policies[policy], original["holdout"][policy])
        track_intervals[policy] = bootstrap(targets, scores, thresholds, plan["bootstrap_replicates"],
                                            plan["seed"] + offset)
        assert_matches(track_intervals[policy], original["confidence_intervals"][policy])
        cluster_intervals[policy] = bootstrap(targets, scores, thresholds, plan["bootstrap_replicates"],
                                              plan["seed"] + offset, groups)
    blind = blind_summary(plan, rows, root)
    for key in ("overall", "by_label"):
        assert_matches(blind[key], archived_blind[key])
    source_rows = {r["track_id"]: r for r in manifest["tracks"]}
    historical_artists = {source_rows[i]["artist_id"] for i in plan["historical_test_ids"]}
    learning = read_json(root / "data/sample_manifest.json")["samples"]
    learning_ids = {r["track_id"] for r in learning}
    learning_artists = {r["artist_id"] for r in learning}
    ids = np.array([r["track_id"] for r in rows])
    sensitivities = {}
    for name, mask in (("exclude_learning_tracks", ~np.isin(ids, list(learning_ids))),
                       ("exclude_learning_artists", ~np.isin(groups, list(learning_artists)))):
        sensitivities[name] = dict(tracks=int(mask.sum()), artists=len(np.unique(groups[mask])),
            excluded_ids=ids[~mask].tolist(), policies={policy: evaluate(targets[mask], scores[mask], thresholds)
                                                     for policy, thresholds in plan["thresholds"].items()})
    return dict(scope="post-hoc audit of the existing holdout; no fitting or new independent evaluation",
                baseline_commit=baseline["commit"], protected_files_verified=len(baseline["sha256"]),
                input_sha256={name: sha256(root / name) for name in (
                    "data/evaluation/holdout.json", "experiments/final_holdout_plan.json",
                    "data/final_blind_review.csv", "data/evaluation/original_metrics.json")},
                original_metrics_verified=True, original_track_intervals_verified=True,
                original_blind_summary_verified=True, policies=policies,
                track_intervals=track_intervals, artist_cluster_intervals=cluster_intervals,
                cluster_rule="Sample 89 artists uniformly with replacement; retain all their tracks each time. "
                    "Metrics pool tracks (not equal artist weighting); sample track count can vary. "
                    "No-positive AP/recall and no-prediction precision are zero, matching archived conventions. "
                    "All replicates retained; counts reported. Fixed model only, no training uncertainty.",
                boundary=dict(holdout_tracks=len(rows), holdout_artists=len(np.unique(groups)),
                    historical_artist_overlap=len(set(groups) & historical_artists),
                    tracks_on_historical_artists=int(np.isin(groups, list(historical_artists)).sum()),
                    largest_artist_cluster=max(int((groups == g).sum()) for g in np.unique(groups)),
                    learning_track_overlap=sorted(set(ids) & learning_ids)),
                sensitivities=sensitivities, listening=blind)


def render_markdown(result):
    lines = ["# Evaluation review supplement", "", "Added 2026-09-12. This is a post-hoc analysis of saved predictions.",
             "The original 239-track results remain the primary historical record. No model was fitted",
             "or tuned for this analysis, and the reduced subsets are not new independent tests.", "",
             "Recompute with `python evaluate_release.py --output outputs/review_audit.json`.",
             "All original metrics for both policies, their track-bootstrap intervals, and the",
             "overall/per-label listening counts were checked against the published archived results.", "",
             "## Historical exposure", "",
             "The 239 holdout tracks cover 89 artists excluded from final fitting. Of these,",
             "40 artists also occurred in the previously observed historical test, accounting for",
             "150 holdout tracks. `track_0543400` was a learning/decoding sample before final evaluation.",
             "The original 480-track subset excluded all three learning tracks; expansion reintroduced",
             "this one. The original dataset document describes that earlier subset only.", "",
             "## Artist-cluster uncertainty", "", result["cluster_rule"], "",
             "Each policy uses 2,000 replicates and percentile 95% intervals; seeds are 20260830",
             "(primary) and 20260831 (secondary). Artist IDs are sorted before seeded sampling.",
             "This supplements the final protocol's track bootstrap and documents a change from",
             "the earlier expansion protocol's promised artist-grouped uncertainty. It does not",
             "address source-label errors, pretraining overlap, or selection of this convenience subset.", "",
             "| Policy / metric | Estimate | Original track interval | Artist-cluster interval |",
             "| --- | ---: | --- | --- |"]
    for policy in ("f1", "precision_target"):
        for key in ("micro_precision", "micro_recall", "micro_f1", "macro_ap", "track_output_coverage"):
            a = result["track_intervals"][policy]["overall"][key]
            b = result["artist_cluster_intervals"][policy]["overall"][key]
            lines.append(f"| {policy} / {key} | {result['policies'][policy][key]:.3f} | {a[0]:.3f}–{a[1]:.3f} | {b[0]:.3f}–{b[1]:.3f} |")
    lines += ["", "## Previously handled sample: sensitivity", "",
              "Both exclusions were chosen after the historical evaluation was available. All four",
              "labels, thresholds, model scores and heads stay fixed. Results below use the primary policy;",
              "the machine-readable supplement also includes the secondary policy.", "",
              "| Subset | Tracks | Artists | Micro precision | Micro recall | Micro F1 | Macro AP |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    subsets = [("Original", 239, 89, result["policies"]["f1"])]
    subsets += [(name, r["tracks"], r["artists"], r["policies"]["f1"]) for name, r in result["sensitivities"].items()]
    for name, count, artists, r in subsets:
        lines.append(f"| {name} | {count} | {artists} | {r['micro_precision']:.3f} | {r['micro_recall']:.3f} | {r['micro_f1']:.3f} | {r['macro_ap']:.3f} |")
    lines += ["", "These exclusions cannot remove the broader historical artist exposure or establish",
              "that earlier handling had no influence on project decisions.", "", "## Listening order", "",
              "The original interface hid source targets and model results, but each label's five",
              "source-positive questions preceded its five source-negative questions. Someone familiar",
              "with the protocol could infer the source target from order. This is an order-structured",
              "single-reviewer audit, not a fully concealed blind test. The recorded 29/36 agreement",
              "remains descriptive; actual influence from order cannot be established from these records.",
              "Future audits should shuffle new questions globally with a recorded seed before listening.",
              "The historical query order and human answers have not been rewritten.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional generated JSON supplement")
    parser.add_argument("--markdown", type=Path, help="Optional generated Markdown supplement")
    args = parser.parse_args()
    result = analyze()
    for path, content in ((args.output, json.dumps(result, indent=2) + "\n"),
                          (args.markdown, render_markdown(result))):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    print("Verified original metrics, track intervals, listening counts and historical protected files.")
    print(json.dumps({"boundary": result["boundary"], "sensitivities": {
        name: {"tracks": r["tracks"], "artists": r["artists"],
               "primary_micro_f1": r["policies"]["f1"]["micro_f1"],
               "primary_macro_ap": r["policies"]["f1"]["macro_ap"]}
        for name, r in result["sensitivities"].items()}}, indent=2))


if __name__ == "__main__":
    main()
