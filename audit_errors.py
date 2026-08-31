"""Reproduce an exploratory error audit on the reused development validation set.

This script does not listen to audio, relabel examples, tune thresholds, or touch
the test split.  The neighbour vocabulary only flags source-tag disagreements
that deserve human listening; it is not used as ground truth.
"""

from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
LABELS = ["electronic", "pop", "ambient", "rock"]
NEIGHBOURS = {
    "electronic": {"dance", "downtempo", "drumnbass", "dubstep", "house", "idm", "techno", "trance", "triphop"},
    "pop": {"chanson", "indie", "instrumentalpop", "popfolk", "poprock"},
    "ambient": {"atmospheric", "chillout", "easylistening", "lounge", "newage"},
    "rock": {"alternative", "hardrock", "instrumentalrock", "metal", "poprock", "postrock", "rocknroll"},
}


def load_inputs():
    manifest = json.loads((ROOT / "data/dataset_manifest.json").read_text())
    metrics = json.loads((ROOT / "outputs/improvement/mert_metrics.json").read_text())
    with np.load(ROOT / "outputs/improvement/mert_scores.npz", allow_pickle=False) as data:
        ids = data["ids"]
        y = data["y"]
        scores = data["validation_scores"]
    validation_ids = ids[-len(scores):]
    rows_by_id = {row["track_id"]: row for row in manifest["tracks"]}
    rows = [rows_by_id[str(track_id)] for track_id in validation_ids]
    if any(row["split"] != "validation" for row in rows):
        raise ValueError("Audit input includes a non-validation track.")
    if not np.array_equal(y[-len(rows):], np.array([row["targets"] for row in rows])):
        raise ValueError("Saved labels differ from the frozen manifest.")
    thresholds = np.array(metrics["thresholds"]["f1"])
    return rows, y[-len(rows):], scores, thresholds


def audit():
    rows, y, scores, thresholds = load_inputs()
    predicted = scores >= thresholds
    details = []
    summary = []
    for j, label in enumerate(LABELS):
        fp = np.flatnonzero(predicted[:, j] & (y[:, j] == 0))
        fn = np.flatnonzero(~predicted[:, j] & (y[:, j] == 1))
        fp = fp[np.argsort(-scores[fp, j], kind="stable")]
        fn = fn[np.argsort(scores[fn, j], kind="stable")]
        neighbour_count = 0
        for kind, indices in (("false_positive", fp), ("false_negative", fn)):
            for rank, i in enumerate(indices, 1):
                source_tags = set(rows[i]["tags"])
                neighbour_tags = sorted(source_tags & NEIGHBOURS[label]) if kind == "false_positive" else []
                if neighbour_tags:
                    neighbour_count += 1
                details.append({
                    "label": label,
                    "error_type": kind,
                    "severity_rank": rank,
                    "selected_for_listening": rank <= 3,
                    "track_id": rows[i]["track_id"],
                    "artist": rows[i]["artist"],
                    "title": rows[i]["title"],
                    "score": round(float(scores[i, j]), 6),
                    "threshold": round(float(thresholds[j]), 6),
                    "source_tags": rows[i]["tags"],
                    "heuristic_neighbour_tags": neighbour_tags,
                    "audio_file": rows[i]["audio_file"],
                    "track_url": rows[i]["track_url"],
                })
        summary.append({
            "label": label,
            "false_positives": int(len(fp)),
            "false_negatives": int(len(fn)),
            "false_positives_with_neighbour_tag": neighbour_count,
        })
    return summary, details


def write_outputs():
    summary, details = audit()
    out = ROOT / "outputs/improvement"
    out.mkdir(parents=True, exist_ok=True)
    (out / "error_audit.json").write_text(json.dumps({
        "scope": "exploratory audit after development validation results were known",
        "ground_truth_warning": "Source tags and heuristic neighbours do not prove auditory correctness.",
        "labels": LABELS,
        "heuristic_neighbours": {key: sorted(value) for key, value in NEIGHBOURS.items()},
        "summary": summary,
        "cases": details,
    }, ensure_ascii=False, indent=2) + "\n")
    fields = ["label", "error_type", "severity_rank", "selected_for_listening", "track_id",
              "artist", "title", "score", "threshold", "source_tags", "heuristic_neighbour_tags",
              "audio_file", "track_url"]
    with (out / "error_audit.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in details:
            writer.writerow({**row, "source_tags": ";".join(row["source_tags"]),
                             "heuristic_neighbour_tags": ";".join(row["heuristic_neighbour_tags"])})
    selected = [row for row in details if row["selected_for_listening"]]
    counts = Counter((row["label"], row["error_type"]) for row in selected)
    if any(counts[label, kind] != 3 for label in LABELS for kind in ("false_positive", "false_negative")):
        raise ValueError("Expected exactly three listening cases per label and error type.")
    return summary, selected


if __name__ == "__main__":
    summary, selected = write_outputs()
    print(json.dumps({"summary": summary, "listening_cases": len(selected)}, indent=2))
