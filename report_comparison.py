"""Report measured validation results, threshold diagnostics and error cases."""

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
from sklearn.metrics import f1_score

from compare_features import OUTPUT, PLAN_PATH, check_protected


def read_predictions(name, labels):
    with (OUTPUT / f"{name}_validation_predictions.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    truth = np.array([[int(r[f"true_{tag}"]) for tag in labels] for r in rows])
    scores = np.array([[float(r[f"score_{tag}"]) for tag in labels] for r in rows])
    return rows, truth, scores


def main():
    check_protected(json.loads(PLAN_PATH.read_text()))
    metrics = json.loads((OUTPUT / "metrics.json").read_text())
    labels, models = metrics["labels"], metrics["models"]
    manifest = json.loads((ROOT / "data/dataset_manifest.json").read_text())
    source = {r["track_id"]: r for r in manifest["tracks"]}
    predictions = {name: read_predictions(name, labels) for name in models}
    base, extended = models["mfcc"]["validation"], models["extended"]["validation"]
    lines = ["# Validation-only feature comparison", "", "Run date: 2026-08-28.", "",
             "Historical exact-source-tag experiment. Its 300/90 development split and validation-tuned thresholds differ from the expanded broad-label comparisons in [README.md](README.md#model-development-path).", "",
             "## Scope", "",
             "One fixed comparison: 300 training tracks and 90 validation tracks; artist IDs are separated.",
             "The original baseline test results were already known. No test audio was read or predicted in this experiment.",
             "This is exploratory development evidence, not a new blind test. The default model and original results are unchanged.",
             "See the [pre-run protocol](experiments/PROTOCOL.md) and [frozen plan](experiments/feature_comparison_plan.json).", "",
             "## Measured comparison", "",
             "| Input | Features | Validation Macro AP (primary) | Validation Macro-F1 | Validation Micro-F1 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, title in [("mfcc", "MFCC statistics"), ("extended", "MFCC + spectral/onset/chroma statistics")]:
        row = models[name]
        score = row["validation"]
        lines.append(f'| {title} | {row["n_features"]} | {score["macro_ap"]:.4f} | {score["macro_f1"]:.4f} | {score["micro_f1"]:.4f} |')
    lines += ["", f'Macro AP changed by {extended["macro_ap"]-base["macro_ap"]:+.4f}; Macro-F1 changed by {extended["macro_f1"]-base["macro_f1"]:+.4f}.',
              "AP evaluates ranking without choosing a tag cutoff. F1 combines precision and recall after thresholding.",
              "Both models chose thresholds on these same validation tracks; their F1 values are optimistic development measurements.",
              "Do not compare this table with the 90-track test scores in RESULTS.md as if the rows came from the same evaluation set.", "",
              "| Tag | Positives | MFCC AP | Extended AP | MFCC F1 | Extended F1 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for a, b in zip(base["per_label"], extended["per_label"], strict=True):
        lines.append(f'| {a["label"]} | {a["support"]} | {a["average_precision"]:.4f} | {b["average_precision"]:.4f} | {a["f1"]:.4f} | {b["f1"]:.4f} |')
    lines += ["", "![Validation comparison and pop threshold curve on the original 90 validation tracks](docs/assets/feature_comparison.png)",
              "", "## Thresholds and errors", "",
              "TP: predicted and annotated positive; FP: predicted positive but unannotated; FN: annotated positive but missed; TN: neither.",
              "These counts are relative to dataset annotations, which may be incomplete.", "",
              "| Input | Tag | Selected threshold | TP | FP | FN | TN |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in models.items():
        for tag, threshold, count in zip(labels, result["thresholds"], result["counts"], strict=True):
            lines.append(f'| {name} | {tag} | {threshold:.2f} | {count["tp"]} | {count["fp"]} | {count["fn"]} | {count["tn"]} |')
    lines += ["", "With every threshold fixed at 0.5 (a diagnostic, not a new model):", "",
              "| Input | Validation Macro-F1 | Validation Micro-F1 |", "| --- | ---: | ---: |"]
    for name, result in models.items():
        score = result["fixed_05"]
        lines.append(f'| {name} | {score["macro_f1"]:.4f} | {score["micro_f1"]:.4f} |')
    lines += ["", "Lowering a fixed tag's threshold can add both true positives and false positives; it does not retrain the classifier.",
              "The grid searches tag F1, not a requirement for high precision. A threshold below 0.5 is not itself a bug.", "",
              "## Two baseline validation error cases", "",
              "Selected by descending number of mismatched tags, then track ID. No auditory cause has been established.", ""]
    rows, truth, scores = predictions["mfcc"]
    guessed = scores >= np.array(models["mfcc"]["thresholds"])
    mismatch = (truth != guessed).sum(axis=1)
    selected = sorted(range(len(rows)), key=lambda i: (-mismatch[i], rows[i]["track_id"]))[:2]
    cases = []
    for i in selected:
        track_id = rows[i]["track_id"]
        if mismatch[i] == 0:
            continue
        true_tags = [tag for tag, flag in zip(labels, truth[i], strict=True) if flag]
        predicted_tags = [tag for tag, flag in zip(labels, guessed[i], strict=True) if flag]
        lines += [f'### {track_id}', "", f'[Original track]({source[track_id]["track_url"]}); validation partition.',
                  f'Dataset target tags: {", ".join(true_tags) or "none"}. Predicted: {", ".join(predicted_tags) or "none"}.', "",
                  "| Tag | Annotated | Score | Threshold | Selected |", "| --- | --- | ---: | ---: | --- |"]
        for j, tag in enumerate(labels):
            lines.append(f'| {tag} | {bool(truth[i,j])} | {scores[i,j]:.4f} | {models["mfcc"]["thresholds"][j]:.2f} | {bool(guessed[i,j])} |')
        cases.append(dict(track_id=track_id, split="validation", true_tags=true_tags,
                          predicted_tags=predicted_tags, scores=scores[i].tolist()))
        lines.append("")
    lines += ["## Decision and limits", "",
              "The predeclared primary metric increased slightly, while both F1 averages decreased. There is no consistent improvement across metrics or tags.",
              "Keep the simpler 26-feature baseline as the default demo, as planned; retain the candidate and this mixed result as a documented comparison.",
              "This is a simplicity/development decision, not a claim that MFCC wins the primary metric or that either method is statistically superior.",
              "No confidence interval or significance claim is made. The validation set is small, coverage-enriched and reused for threshold selection.",
              "Possible limitations include lost time order, redundant descriptors, key-sensitive chroma and noisy track-level labels. This experiment does not isolate their causal contributions.",
              "Adding these descriptors does not provide tempo estimation, chord recognition, or a full description of rhythm and harmony.",
              "No additional hyperparameter search or test-set evaluation was performed. The experiment stops after this fixed comparison.", "",
              "## Reproduce locally", "", "```bash", ".venv/bin/python compare_features.py", ".venv/bin/python report_comparison.py", "```", "",
              "Requires the original local baseline artifacts and prepared audio. No new packages or data downloads are required in the existing environment.",
              "Raw metrics (`outputs/feature_comparison/metrics.json`, generated locally) · MFCC predictions (`outputs/feature_comparison/mfcc_validation_predictions.csv`, generated locally) · Extended predictions (`outputs/feature_comparison/extended_validation_predictions.csv`, generated locally)", "",
              "[Metric definitions: F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) and [Average Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).", ""]
    (ROOT / "EXPERIMENTS.md").write_text("\n".join(lines))
    (OUTPUT / "error_cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    styles = [("mfcc", "MFCC (26)", "#146e87", -0.18), ("extended", "Extended (46)", "#b85e21", 0.18)]
    positions = np.arange(4)
    grid = np.linspace(.1, .9, 17)
    sweep = []
    pop = labels.index("pop")
    for name, title, color, offset in styles:
        bars = axes[0].bar(positions + offset, [r["average_precision"] for r in models[name]["validation"]["per_label"]], .34, color=color, label=title)
        axes[0].bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        _, truth, scores = predictions[name]
        values = [f1_score(truth[:, pop], scores[:, pop] >= threshold, zero_division=0) for threshold in grid]
        axes[1].plot(grid, values, marker="o", markersize=3, color=color, label=title)
        chosen = models[name]["thresholds"][pop]
        chosen_f1 = models[name]["validation"]["per_label"][pop]["f1"]
        axes[1].scatter([chosen], [chosen_f1], color=color, s=80, marker="*", zorder=4)
        for threshold, f1 in zip(grid, values, strict=True):
            sweep.append(dict(model=name, label="pop", threshold=float(threshold), f1=float(f1), selected=bool(threshold == chosen)))
    axes[0].set(xticks=positions, xticklabels=labels, ylim=(0, 1), ylabel="Average Precision", title="Ranking by tag (higher is better)")
    axes[1].set(xlim=(.08, .92), ylim=(0, .6), xlabel="Tag threshold", ylabel="F1", title="Pop: threshold diagnostic (* = chosen)")
    for ax in axes:
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    figure.suptitle("Development comparison · 90 validation tracks (not test results)", fontsize=12)
    figure.savefig(OUTPUT / "comparison.png", dpi=160)
    plt.close(figure)
    assets = ROOT / "docs/assets"
    assets.mkdir(parents=True, exist_ok=True)
    copyfile(OUTPUT / "comparison.png", assets / "feature_comparison.png")
    (OUTPUT / "threshold_sweep.json").write_text(json.dumps(sweep, indent=2) + "\n")
    check_protected(json.loads(PLAN_PATH.read_text()))
    print("Saved EXPERIMENTS.md, comparison.png, threshold sweep and two validation error cases.")


if __name__ == "__main__":
    main()
