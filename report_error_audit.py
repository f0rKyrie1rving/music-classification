"""Render the machine-readable validation error audit as a learning report."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    audit = json.loads((ROOT / "outputs/improvement/error_audit.json").read_text())
    lines = [
        "# Development validation error audit",
        "",
        "This is an **exploratory** audit made after the 90-track development validation results were known. "
        "It is useful for diagnosing the next data/model step, not for making a fresh performance claim.",
        "",
        "## What was counted",
        "",
        "The table uses the MERT candidate's F1-oriented thresholds. A false positive means the model emitted a "
        "label absent from the source tags; a false negative means a source target tag did not pass its threshold.",
        "",
        "| Label | False positives | False negatives | FPs with a declared neighbouring tag |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in audit["summary"]:
        lines.append(f"| {row['label']} | {row['false_positives']} | {row['false_negatives']} | "
                     f"{row['false_positives_with_neighbour_tag']} |")
    lines += [
        "",
        "Ten of 56 false positives contain one of the declared neighbouring tags. Examples include `hardrock` "
        "without `rock`, `rocknroll` without `rock`, and `chanson` without `pop`. This is evidence of a taxonomy "
        "or annotation issue worth listening to; it does **not** prove that the model is correct. The heuristic "
        "mapping is stored in `outputs/improvement/error_audit.json` and is never used to change labels or metrics.",
        "",
        "## Fixed listening sample",
        "",
        "For each label, take the three highest-scoring false positives and three lowest-scoring false negatives. "
        "This produces 24 cases without hand-picking appealing examples.",
        "",
        "| Label | Error | Track | Artist — title | Score / threshold | Source tags | Neighbour flag |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    selected = [row for row in audit["cases"] if row["selected_for_listening"]]
    for row in selected:
        audio = row["audio_file"].replace(" ", "%20")
        tags = ", ".join(row["source_tags"])
        neighbour = ", ".join(row["heuristic_neighbour_tags"]) or "—"
        error = "FP" if row["error_type"] == "false_positive" else "FN"
        title = str(row["title"]).replace("|", "\\|")
        artist = str(row["artist"]).replace("|", "\\|")
        lines.append(f"| {row['label']} | {error} | [{row['track_id']}]({audio}) | {artist} — {title} | "
                     f"{row['score']:.3f} / {row['threshold']:.3f} | {tags} | {neighbour} |")
    lines += [
        "",
        "## How to listen",
        "",
        "Open `data/error_listening_review.csv`. For each case, listen without treating the model prediction as an "
        "answer. Enter `yes`, `no`, or `uncertain` under `auditor_hears_label`, then add one short reason. "
        "A single listener's judgement remains a portfolio error analysis, not replacement benchmark ground truth.",
        "",
        "## What this changes",
        "",
        "The automated findings justify two actions already frozen in the expansion protocol: use all eligible "
        "training data instead of resampling a small subset, and keep the new test tracks sealed until every model "
        "choice is fixed. No label has been changed as a result of this audit.",
        "",
    ]
    (ROOT / "ERROR_AUDIT.md").write_text("\n".join(lines))

    fields = ["label", "error_type", "track_id", "artist", "title", "score", "threshold",
              "source_tags", "audio_file", "auditor_hears_label", "auditor_reason"]
    path = ROOT / "data/error_listening_review.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({
                "label": row["label"], "error_type": row["error_type"], "track_id": row["track_id"],
                "artist": row["artist"], "title": row["title"], "score": row["score"],
                "threshold": row["threshold"], "source_tags": ";".join(row["source_tags"]),
                "audio_file": row["audio_file"], "auditor_hears_label": "", "auditor_reason": "",
            })
    print(f"Wrote ERROR_AUDIT.md and {len(selected)} listening rows.")


if __name__ == "__main__":
    main()
