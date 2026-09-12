"""Render the compact holdout-results figure used on the project landing page."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "data/final_metrics_summary.json"
OUTPUT = ROOT / "docs/assets/final_holdout_results.png"


def main():
    metrics = json.loads(METRICS.read_text())
    rows, labels = metrics["per_label"], metrics["labels"]
    x = np.arange(len(labels)); width = 0.23
    colors = {"Precision": "#31688e", "Recall": "#35b779", "F1": "#fde725"}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for offset, key in zip((-width, 0, width), ("precision", "recall", "f1"), strict=True):
        name = key.capitalize()
        axes[0].bar(x + offset, [row[key] for row in rows], width,
                    label=name, color=colors[name])
    axes[0].set(title="Per-label performance", ylabel="Score", xticks=x,
                xticklabels=labels, ylim=(0, 1))
    axes[0].grid(axis="y", alpha=0.2); axes[0].legend(frameon=False, ncols=3)

    names = ["Micro\nprecision", "Micro\nrecall", "Micro\nF1", "Macro\nAP"]
    values = [metrics["micro_precision"], metrics["micro_recall"],
              metrics["micro_f1"], metrics["macro_ap"]]
    bars = axes[1].bar(names, values, color=["#31688e", "#35b779", "#fde725", "#6a51a3"])
    axes[1].bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4)
    axes[1].set(title="Primary four-label policy", ylabel="Score", ylim=(0, 1))
    axes[1].grid(axis="y", alpha=0.2)

    figure.suptitle("Discogs-MAEST final holdout evaluation", fontsize=15, weight="bold")
    figure.text(0.5, -0.015,
                "239 tracks · 89 artists excluded from fitting · prior exposure disclosed in report",
                ha="center", color="#555555")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
