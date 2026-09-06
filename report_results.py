"""Render the baseline metrics report and two representative errors."""

import csv
import json
import os
from pathlib import Path
from shutil import copyfile

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    output = ROOT / "outputs"
    metrics = json.loads((output / "baseline_metrics.json").read_text())
    manifest = json.loads((ROOT / "data/dataset_manifest.json").read_text())
    by_id = {r["track_id"]: r for r in manifest["tracks"]}
    labels, test = metrics["labels"], metrics["test"]
    prior = np.mean([r["targets"] for r in manifest["tracks"] if r["split"] == "train"], axis=0)
    prior_tags = [t for t, score, threshold in zip(labels, prior, metrics["prior_thresholds"], strict=True)
                  if score >= threshold]
    lines = ["# First MFCC baseline", "", "Actual results on the custom MTG-Jamendo subset.", "",
             "Historical exact-source-tag experiment: the 90 test tracks here are distinct from the 90 development-validation tracks in later early-stage reports. The final broad-label model is evaluated separately in [FINAL_RESULTS.md](FINAL_RESULTS.md).", "",
             "## Protocol", "",
             f'Train / validation / test: {metrics["split_counts"]["train"]} / {metrics["split_counts"]["validation"]} / {metrics["split_counts"]["test"]} tracks.',
             "Artist IDs do not overlap across partitions. The model uses 13 MFCC means and 13 standard deviations.",
             "Standardization is fitted on training data only. Each tag threshold is selected on validation data, then fixed for testing.",
             "The comparison baseline predicts the training-set tag frequencies and uses the same validation threshold procedure.", "",
             "## Test results", "", "| Method | Micro-F1 | Macro-F1 | Macro AP |", "| --- | ---: | ---: | ---: |"]
    for name, title in [("prior", "Tag-frequency baseline"), ("mfcc", "MFCC + logistic regression")]:
        r = test[name]
        lines.append(f'| {title} | {r["micro_f1"]:.3f} | {r["macro_f1"]:.3f} | {r["macro_ap"]:.3f} |')
    lines += ["", "| Tag | Test positives | Precision | Recall | F1 | AP |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in test["mfcc"]["per_label"]:
        lines.append(f'| {r["label"]} | {r["support"]} | {r["precision"]:.3f} | {r["recall"]:.3f} | {r["f1"]:.3f} | {r["average_precision"]:.3f} |')
    difference = test["mfcc"]["macro_f1"] - test["prior"]["macro_f1"]
    lines += ["", "### Interpretation", "",
              f'The observed Macro-F1 difference is {difference:+.3f}. Macro-F1 averages the four per-tag F1 scores; it is not classification accuracy.',
              "F1 balances precision and recall; AP summarizes the ranking of positive examples across thresholds.",
              "Per-tag results matter: an average improvement does not mean every tag improved. No statistical significance or population-wide gain is claimed.",
              "The frequency baseline assigns the same score to every track for a given tag; it does not use audio.",
              f'After validation threshold selection, it always outputs: {", ".join(prior_tags) or "no tags"}.', "",
              "| Tag | Validation-selected threshold |", "| --- | ---: |"]
    for label, threshold in zip(labels, metrics["thresholds"], strict=True):
        lines.append(f'| {label} | {threshold:.2f} |')
    figure, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    x = np.arange(len(labels))
    for shift, name, title, color in [(-0.18, "prior", "Tag-frequency baseline", "#9ba3aa"),
                                      (0.18, "mfcc", "MFCC + logistic regression", "#146e87")]:
        bars = ax.bar(x + shift, [r["f1"] for r in test[name]["per_label"]], 0.34, label=title, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set(xticks=x, xticklabels=labels, ylabel="F1 score", ylim=(0, 1.15),
           title=f'First baseline · {metrics["split_counts"]["test"]} test tracks')
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    figure.savefig(output / "baseline_f1.png", dpi=160)
    plt.close(figure)
    assets = ROOT / "docs/assets"
    assets.mkdir(parents=True, exist_ok=True)
    copyfile(output / "baseline_f1.png", assets / "baseline_f1.png")
    lines += ["", "![Per-tag F1 on the original 90-track test set](docs/assets/baseline_f1.png)", "", "## Two error examples", "",
              "Selected by the number of mismatched tags (descending), then track ID; all predictions remain available in the CSV.", ""]
    with (output / "test_predictions.csv").open(newline="") as file:
        predictions = list(csv.DictReader(file))
    errors = []
    for row in predictions:
        truth = [int(row[f"true_{t}"]) for t in labels]
        predicted = [float(row[f"score_{t}"]) >= threshold for t, threshold in zip(labels, metrics["thresholds"], strict=True)]
        count = sum(a != b for a, b in zip(truth, predicted, strict=True))
        if count:
            errors.append((count, row["track_id"], truth, predicted))
    for _, track_id, truth, predicted in sorted(errors, key=lambda r: (-r[0], r[1]))[:2]:
        original = ", ".join(t for t, flag in zip(labels, truth, strict=True) if flag) or "none of the four targets"
        guess = ", ".join(t for t, flag in zip(labels, predicted, strict=True) if flag) or "none of the four targets"
        lines.append(f'- [{track_id}]({by_id[track_id]["track_url"]}): dataset tags = {original}; predicted = {guess}.')
    lines += ["", "## Limitations", "",
              "This is a small, coverage-enriched subset of four archive shards, not the official benchmark distribution.",
              "Uploaders' tags may be incomplete; a missing tag is not proof that a musical quality is absent.",
              "Only the first 30 seconds are analyzed. MFCC summary statistics discard time order and do not fully describe rhythm or harmony.",
              "Scores are not calibrated confidence. Results do not establish performance on arbitrary commercial music.",
              "No MFCC-count or classifier hyperparameter search has been performed. Test results were not used to tune this baseline.", "",
              "See [dataset protocol and licenses](data/DATASET.md), raw metrics (`outputs/baseline_metrics.json`, generated locally), and all test predictions (`outputs/test_predictions.csv`, generated locally).", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    print("Saved RESULTS.md, outputs/baseline_f1.png, and docs/assets/baseline_f1.png.")


if __name__ == "__main__":
    main()
