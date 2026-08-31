"""Compare the frozen blind review with source targets and final decisions."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PLAN = ROOT / "experiments/final_holdout_plan.json"
REVIEW = ROOT / "data/final_blind_review.csv"
PREDICTIONS = ROOT / "outputs/final_maest/holdout_predictions.csv"
OUT_JSON = ROOT / "outputs/final_maest/blind_review_results.json"
OUT_CSV = ROOT / "outputs/final_maest/blind_review_comparison.csv"
LABELS = ("electronic", "pop", "ambient", "rock")
DECISIONS = {"yes": 1, "no": 0, "uncertain": None}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(rows):
    """Summarize a deliberately balanced query set without claiming prevalence."""
    decisions = Counter(row["human_decision"] for row in rows)
    decisive = [row for row in rows if row["human_target"] is not None]
    source_matches = sum(row["human_target"] == row["source_target"] for row in decisive)
    model_matches = sum(row["human_target"] == row["model_prediction"] for row in decisive)
    tp = sum(row["human_target"] == 1 and row["model_prediction"] == 1 for row in decisive)
    fp = sum(row["human_target"] == 0 and row["model_prediction"] == 1 for row in decisive)
    fn = sum(row["human_target"] == 1 and row["model_prediction"] == 0 for row in decisive)
    tn = sum(row["human_target"] == 0 and row["model_prediction"] == 0 for row in decisive)

    strata = {}
    for source_target in (1, 0):
        selected = [row for row in rows if row["source_target"] == source_target]
        counts = Counter(row["human_decision"] for row in selected)
        strata[str(source_target)] = {
            "queries": len(selected),
            "human_yes": counts["yes"],
            "human_no": counts["no"],
            "human_uncertain": counts["uncertain"],
        }

    return {
        "queries": len(rows),
        "human_yes": decisions["yes"],
        "human_no": decisions["no"],
        "human_uncertain": decisions["uncertain"],
        "decisive_queries": len(decisive),
        "source_human_agreement_count": source_matches,
        "source_human_agreement_rate_decisive": ratio(source_matches, len(decisive)),
        "model_human_agreement_count": model_matches,
        "model_human_agreement_rate_decisive": ratio(model_matches, len(decisive)),
        "model_source_agreement_count": sum(
            row["model_prediction"] == row["source_target"] for row in rows
        ),
        "model_source_agreement_rate": ratio(
            sum(row["model_prediction"] == row["source_target"] for row in rows), len(rows)
        ),
        "model_human_confusion_decisive": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "model_human_precision_decisive": ratio(tp, tp + fp),
        "model_human_recall_decisive": ratio(tp, tp + fn),
        "model_human_f1_decisive": ratio(2 * tp, 2 * tp + fp + fn),
        "source_strata": strata,
    }


def analyze():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    queries = plan["blind_review_queries"]
    if (len(queries) != 40 or len({item["track_id"] for item in queries}) != 40
            or Counter(item["label"] for item in queries) != Counter({label: 10 for label in LABELS})):
        raise ValueError("Frozen blind-query plan is not the expected balanced 40-query set.")

    with REVIEW.open(newline="", encoding="utf-8") as file:
        reviews = list(csv.DictReader(file))
    by_query = {int(row["query_id"]): row for row in reviews}
    if len(reviews) != 40 or len(by_query) != 40:
        raise ValueError("Blind-review sheet must contain 40 unique query IDs.")

    with PREDICTIONS.open(newline="", encoding="utf-8") as file:
        predictions = {row["track_id"]: row for row in csv.DictReader(file)}

    rows = []
    for query in queries:
        review = by_query.get(query["query_id"])
        prediction = predictions.get(query["track_id"])
        if review is None or prediction is None:
            raise ValueError(f"Missing review or prediction for query {query['query_id']}.")
        if (review["label"] != query["label"]
                or review["track_id"] != query["track_id"]):
            raise ValueError(f"Frozen query {query['query_id']} changed in the review sheet.")
        decision = review["auditor_hears_label"].strip()
        if decision not in DECISIONS:
            raise ValueError(f"Query {query['query_id']} has no valid completed answer.")
        label = query["label"]
        source_target = int(query["source_target"])
        if int(prediction[f"source_{label}"]) != source_target:
            raise ValueError(f"Source target changed for query {query['query_id']}.")
        human_target = DECISIONS[decision]
        model_prediction = int(prediction[f"predicted_f1_{label}"])
        rows.append({
            "query_id": query["query_id"],
            "label": label,
            "track_id": query["track_id"],
            "source_target": source_target,
            "model_score": float(prediction[f"score_{label}"]),
            "model_prediction": model_prediction,
            "human_decision": decision,
            "human_target": human_target,
            "source_human_agree": None if human_target is None else human_target == source_target,
            "model_human_agree": None if human_target is None else human_target == model_prediction,
        })

    result = {
        "purpose": "secondary single-reviewer perceptual check; not model tuning or a population estimate",
        "selection": "five source-positive and five source-negative queries per label, frozen before scoring",
        "uncertain_policy": "exclude uncertain answers from human-agreement denominators",
        "input_sha256": {
            "final_holdout_plan": sha256(PLAN),
            "completed_blind_review": sha256(REVIEW),
            "holdout_predictions": sha256(PREDICTIONS),
        },
        "overall": summarize(rows),
        "by_label": {label: summarize([row for row in rows if row["label"] == label])
                     for label in LABELS},
    }
    return result, rows


def main():
    result, rows = analyze()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = tuple(rows[0])
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    overall = result["overall"]
    print(f"Completed {overall['queries']} queries; {overall['decisive_queries']} decisive.")
    print("Source-human agreement: "
          f"{overall['source_human_agreement_count']}/{overall['decisive_queries']}.")
    print("Model-human agreement: "
          f"{overall['model_human_agreement_count']}/{overall['decisive_queries']}.")


if __name__ == "__main__":
    main()
