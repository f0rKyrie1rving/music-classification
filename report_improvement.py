"""Render the development-diagnostic report from saved metrics."""

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

from improve_model import OUT, load_development


def main():
    load_development()
    result = json.loads((OUT / "mfcc_metrics.json").read_text())
    original = json.loads((ROOT / "outputs/baseline_metrics.json").read_text())["validation"]["mfcc"]
    selected = result["validation"]["f1"]
    grouped = []
    for fraction in (.25, .5, 1.):
        subset = [r for r in result["learning_curve"] if r["fraction"] == fraction]
        grouped.append(dict(fraction=fraction, mean_tracks=float(np.mean([r["fit_tracks"] for r in subset])),
            min_tracks=min(r["fit_tracks"] for r in subset), max_tracks=max(r["fit_tracks"] for r in subset),
            train_mean=float(np.mean([r["train_macro_ap"] for r in subset])),
            held_mean=float(np.mean([r["held_macro_ap"] for r in subset])),
            held_std=float(np.std([r["held_macro_ap"] for r in subset], ddof=0))))
    lines = ["# Performance improvement: development checkpoint", "", "Run date: 2026-08-28.", "",
        "Historical exact-source-tag experiment on 300 training and 90 validation tracks. Its metrics are not directly comparable with the later 903/303 broad-label task; see the [development-stage comparison](README.md#model-development-path).", "",
        "## What ran", "",
        "Three artist-grouped folds on the original 300 training tracks; six predeclared logistic-regression settings.",
        "The scaler is fitted separately inside each fold. Model selection uses mean fold Macro AP; thresholds use out-of-fold training scores.",
        "The existing 90 validation tracks are evaluated without retuning their thresholds in this round. They were used in earlier development, so this is NOT a fresh blind test.",
        "No new test predictions were made. The original baseline model, feature code, manifest and results remain unchanged.",
        "Protocol: [IMPROVEMENT_PROTOCOL.md](experiments/IMPROVEMENT_PROTOCOL.md); exact splits: [improvement_plan.json](experiments/improvement_plan.json).", "",
        "## Bounded parameter comparison", "",
        "C controls regularization strength: a smaller C penalizes large coefficients more strongly. Balanced weighting changes how errors on rare labels affect training.", "",
        "| C | Class weighting | Mean fold Macro AP | Fold 1 | Fold 2 | Fold 3 |",
        "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for trial in result["trials"]:
        c = trial["config"]
        lines.append(f'| {c["C"]} | {c["class_weight"] or "none"} | {trial["mean_macro_ap"]:.4f} | ' +
                     " | ".join(f"{s:.4f}" for s in trial["fold_macro_ap"]) + " |")
    config = result["selected_config"]
    lines += ["", f'Selected C={config["C"]}, class_weight={config["class_weight"]}. This is a bounded development choice, not a statistical superiority claim.',
        "The winning CV score is optimistic after selecting among six settings. Fold-to-fold variation is visible; do not interpret it as a confidence interval.", "",
        "## Learning curve for the original fixed C=1 / balanced setting", "",
        "Nested subsets of training ARTISTS are used inside each fold. The held-out artists stay fixed within that fold.",
        "Macro AP measures ranking, not precision after applying a threshold and not classification accuracy.", "",
        "| Mean training tracks (range) | Training Macro AP | Held-out Macro AP | Held-out fold SD |",
        "| --- | ---: | ---: | ---: |"]
    for g in grouped:
        lines.append(f'| {g["mean_tracks"]:.0f} ({g["min_tracks"]}–{g["max_tracks"]}) | {g["train_mean"]:.4f} | {g["held_mean"]:.4f} | {g["held_std"]:.4f} |')
    lines += ["", "Held-out mean AP improves with more training artists in this small experiment, but one fold is non-monotonic.",
        "This supports investigating data quantity/diversity; it does not establish a sample count that would yield 80% precision.",
        "High training scores on tiny subsets with much lower held-out scores show sensitivity to limited data. The curve does not isolate feature quality from label noise or sampling bias.", "",
        "![Grouped learning curve and ranking on the original 90 validation tracks](docs/assets/mfcc_diagnosis.png)", "",
        "## Existing validation set (90 tracks)", "",
        "| Model / threshold source | Macro AP | Micro-F1 | Macro-F1 |",
        "| --- | ---: | ---: | ---: |",
        f'| Original baseline / same validation set | {original["macro_ap"]:.4f} | {original["micro_f1"]:.4f} | {original["macro_f1"]:.4f} |',
        f'| Selected MFCC head / training OOF | {selected["macro_ap"]:.4f} | {selected["micro_f1"]:.4f} | {selected["macro_f1"]:.4f} |', "",
        "AP comparisons use the same validation tracks and do not depend on thresholds. F1 also reflects the different threshold-selection data, so this is NOT a pure classifier-only F1 comparison.", "",
        "Selected head with training-OOF F1 thresholds:", "",
        "| Tag | Threshold | TP | FP | FN | Precision | Recall | Predictions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for t, r in zip(result["thresholds"]["f1"], selected["per_label"], strict=True):
        lines.append(f'| {r["label"]} | {t:.3f} | {r["tp"]} | {r["fp"]} | {r["fn"]} | {r["precision"]:.3f} | {r["recall"]:.3f} | {r["predictions"]} |')
    lines += ["", f'Micro precision: {selected["micro_precision"]:.3f}; micro recall: {selected["micro_recall"]:.3f}; track output coverage: {selected["track_output_coverage"]:.3f}.',
        "Coverage means at least one of the four tags is emitted; it does not mean the emitted tags are correct.", "",
        "## 80% precision target: NOT met", "",
        "For the selected MFCC head, no tag has a threshold in the predeclared grid that simultaneously meets 80% precision, 30% recall and 10 emitted predictions on the 300 training-OOF tracks.",
        "The strict policy therefore marks all four tags unsupported and emits none. Its precision is undefined (N/A), recall and coverage are zero; stored numeric precision zeros follow the metric library's zero-division convention.",
        "This abstention is a failed operating point, not a successful 80% model and not evidence that all music has no genre.",
        "The result applies to this selected representation/head, this finite threshold grid and this data subset. It does not prove that 80% is impossible with other data or features.", "",
        "## Decision at the MFCC checkpoint", "",
        "Do not promote the selected MFCC head. The ranking improvement on validation is small and the operating trade-off is not adequate for the stated target.",
        "MERT-v0-public is the one planned frozen encoder; its model card lists Music4All and part of FMA for pretraining. Training-corpus overlap with Jamendo has not been audited.",
        "At the initial MFCC checkpoint, model downloads failed and no MERT results were available. This is a historical description of that checkpoint.",
        ("Connectivity has since been resolved using the existing system proxy; the separately completed [MERT comparison](MERT_RESULTS.md) contains its measured results."
         if (OUT / "mert_metrics.json").exists() else
         "The separate MERT comparison is not yet complete; do not substitute paper benchmark numbers for missing project results."),
        "The original test set remains historically observed. Any final performance claim still needs newly preselected independent tracks/artists, a fixed model and thresholds, and uncertainty reporting.", "",
        "## Reproduce the completed diagnostic", "", "```bash", ".venv/bin/python improve_model.py mfcc",
        ".venv/bin/python report_improvement.py", ".venv/bin/python -m unittest discover -s tests -v", "```", "",
        "Requires existing prepared data and baseline artifacts. The split/parameter plan is already frozen; do not rerun the `freeze` action.",
        "Local raw results: metrics (`outputs/improvement/mfcc_metrics.json`, generated locally), scores (`outputs/improvement/mfcc_scores.npz`, generated locally).", "",
        "Sources: [MERT model card](https://huggingface.co/m-a-p/MERT-v0-public), [MERT runtime warning](https://github.com/yizhilll/MERT), [scikit-learn learning curves](https://scikit-learn.org/stable/modules/learning_curve.html).", ""]
    (ROOT / "IMPROVEMENT.md").write_text("\n".join(lines))
    (OUT / "learning_curve_summary.json").write_text(json.dumps(grouped, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), layout="constrained")
    x = [r["mean_tracks"] for r in grouped]
    axes[0].plot(x, [r["train_mean"] for r in grouped], "o-", color="#b96932", label="Training (same tracks)")
    axes[0].errorbar(x, [r["held_mean"] for r in grouped], yerr=[r["held_std"] for r in grouped],
                    fmt="o-", capsize=4, color="#146e87", label="Held-out artists (mean +/- fold SD)")
    axes[0].set(title="More training artists: a small learning curve", xlabel="Mean training tracks per fold", ylabel="Macro Average Precision", ylim=(0, 1.03), xticks=x)
    pos = np.arange(4)
    for offset, values, label, color in [(-.18, original, "Original MFCC", "#767a84"), (.18, selected, "Selected MFCC", "#146e87")]:
        bars = axes[1].bar(pos + offset, [r["average_precision"] for r in values["per_label"]], .34, label=label, color=color)
        axes[1].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes[1].set(title="Existing validation: ranking by tag", ylabel="Average Precision", ylim=(0, 1.03), xticks=pos, xticklabels=result["labels"])
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8, loc="upper center" if ax == axes[1] else "lower right")
    fig.suptitle("Development evidence only — AP is not thresholded precision", fontsize=13)
    fig.savefig(OUT / "diagnosis.png", dpi=160)
    plt.close(fig)
    assets = ROOT / "docs/assets"
    assets.mkdir(parents=True, exist_ok=True)
    copyfile(OUT / "diagnosis.png", assets / "mfcc_diagnosis.png")


if __name__ == "__main__":
    main()
