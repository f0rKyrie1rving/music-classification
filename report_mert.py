"""Report the actual frozen MFCC-vs-MERT development comparison."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from improve_model import OUT, load_development


def main():
    load_development()
    results = {name: json.loads((OUT / f"{name}_metrics.json").read_text()) for name in ("mfcc", "mert")}
    pilot = json.loads((OUT / "mert_pilot_cpu.json").read_text())
    run = json.loads((OUT / "mert_extraction_run.json").read_text())
    a, b = [results[name]["validation"]["f1"] for name in ("mfcc", "mert")]
    precise = results["mert"]["validation"]["precision_target"]
    lines = ["# Frozen music representations: MFCC vs MERT", "", "Development run: 2026-08-28.", "",
        "## Question and scope", "",
        "Does a frozen pretrained music representation improve this small four-genre tagging task over MFCC statistics?",
        "The encoder, pooling, data, classifier grid and selection rule were fixed before viewing MERT outcomes.",
        "See [protocol](experiments/IMPROVEMENT_PROTOCOL.md), [execution settings](experiments/mert_execution_plan.json) and [source review](experiments/MERT_REVIEW.md).", "",
        "Both representations use the same 300 training tracks, the same three artist-grouped folds, six logistic-regression settings, and the same 90 validation tracks.",
        "Choose the head by mean fold Macro AP; select thresholds on its training out-of-fold scores; refit on 300 tracks before validation prediction.",
        "No new test predictions are made. The 90 validation tracks were previously used in development; these are NOT independent final test results.",
        "MERT pretraining used Music4All and part of FMA; overlap with Jamendo has not been audited. Artist separation here applies to our downstream splits, not to the encoder's pretraining corpus.", "",
        "## Representations and runtime", "",
        "MFCC: 13 temporal means plus 13 standard deviations. MERT: resample the prepared first-30-second audio to 16 kHz, split into six 5-second chunks, mean-pool time and 12 transformer layers (excluding layer 0), then average the chunks: 768 features.",
        "The encoder weights are frozen; only the scaler and four linear classifiers are fitted to this project's labels. No layer/crop/pooling search or encoder fine-tuning was performed.",
        "This experimental MERT extractor currently requires at least 30 seconds. The original MFCC CLI still supports its existing shorter-input policy.", "",
        f'CPU pilot: loading {pilot["load_seconds"]:.2f}s; two 30-second extractions {pilot["extraction_seconds"][0]:.2f}s and {pilot["extraction_seconds"][1]:.2f}s; maximum repeated-feature difference {pilot["max_abs_repeat_difference"]:.1g}.',
        f'Full development extraction: {run["total_tracks"]} tracks; {run["extraction_seconds"]:.1f}s in the recorded run ({run["resumed_tracks"]} cached at start).',
        "Platform: Python 3.13.15, Apple Silicon macOS 15.6.1, CPU with four PyTorch threads. GPU and other platforms were not benchmarked.",
        "Runtime dependencies: torch 2.8.0, transformers 4.38.2; exact lock in requirements-mert-lock.txt. The original .venv was not changed.",
        "All downloaded files were verified against the official pinned version. The local checkpoint was read with weights_only=True and strict state-dictionary matching; reviewed custom code runs locally without audio uploads.", "",
        "## Same-protocol comparison", "",
        "F1 policy: per-label thresholds selected for F1 on training OOF predictions.", "",
        "| Representation | Selected C / weight | Mean CV Macro AP | Validation Macro AP | Micro precision | Micro recall | Micro-F1 | Macro-F1 | Track output coverage |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in results.items():
        r, c = result["validation"]["f1"], result["selected_config"]
        cv = next(t["mean_macro_ap"] for t in result["trials"] if t["config"] == c)
        lines.append(f'| {name.upper()} | {c["C"]} / {c["class_weight"] or "none"} | {cv:.4f} | {r["macro_ap"]:.4f} | {r["micro_precision"]:.3f} | {r["micro_recall"]:.3f} | {r["micro_f1"]:.3f} | {r["macro_f1"]:.3f} | {r["track_output_coverage"]:.3f} |')
    lines += ["", f'Validation changes: Macro AP {b["macro_ap"]-a["macro_ap"]:+.4f}, Micro-F1 {b["micro_f1"]-a["micro_f1"]:+.4f}, Macro-F1 {b["macro_f1"]-a["macro_f1"]:+.4f}.',
        "AP measures ranking, not thresholded precision or classification accuracy. CV selection estimates are optimistic after choosing among six settings.",
        "The MFCC row here is the newly selected MFCC head with OOF thresholds, not the original default demo with validation-tuned thresholds.", "",
        "| Tag | Validation positives | MFCC precision | MERT precision | MFCC recall | MERT recall | MFCC AP | MERT AP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for ra, rb in zip(a["per_label"], b["per_label"], strict=True):
        lines.append(f'| {ra["label"]} | {ra["support"]} | {ra["precision"]:.3f} | {rb["precision"]:.3f} | {ra["recall"]:.3f} | {rb["recall"]:.3f} | {ra["average_precision"]:.3f} | {rb["average_precision"]:.3f} |')
    lines += ["", "![Development precision and recall](outputs/improvement/mert_comparison.png)", "",
        "## Precision-oriented operating point", "",
        "Thresholds were chosen only from training OOF scores: >=80% precision, >=30% recall and >=10 emitted predictions per tag, then maximize recall among feasible choices.",
        "Unsupported tags are suppressed under this policy. Precision is N/A when there are no predictions; stored zeros follow the metric function's zero-division convention.", "",
        "| Tag | Feasible on training OOF? | Threshold | Validation TP | FP | FN | Predictions | Precision | Recall |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for supported, t, r in zip(results["mert"]["precision_supported"], results["mert"]["thresholds"]["precision_target"], precise["per_label"], strict=True):
        p = f'{r["precision"]:.3f}' if r["precision_defined"] else "N/A"
        threshold = f"{t:.3f}" if supported else "disabled"
        lines.append(f'| {r["label"]} | {supported} | {threshold} | {r["tp"]} | {r["fp"]} | {r["fn"]} | {r["predictions"]} | {p} | {r["recall"]:.3f} |')
    p_micro = f'{precise["micro_precision"]:.3f}' if sum(r["predictions"] for r in precise["per_label"]) else "N/A"
    lines += ["", f'Micro precision: {p_micro}; micro recall: {precise["micro_recall"]:.3f}; track output coverage: {precise["tracks_with_output"]}/90 = {precise["track_output_coverage"]:.3f}.',
        "The provisional development target additionally requires every tag to have >=80% precision, >=30% recall, >=5 validation predictions, plus >=50% track output coverage. It is not a guarantee of usefulness or a confidence bound.",
        f'**Provisional development target met: {precise["provisional_goal_met"]}.** Do not substitute overall accuracy, AP, or one successful tag for this statement.', "",
        "## MERT's six classifier trials", "",
        "| C | Class weight | Mean CV Macro AP | Fold scores |", "| ---: | --- | ---: | --- |"]
    for t in results["mert"]["trials"]:
        c = t["config"]
        lines.append(f'| {c["C"]} | {c["class_weight"] or "none"} | {t["mean_macro_ap"]:.4f} | ' + ", ".join(f"{v:.4f}" for v in t["fold_macro_ap"]) + " |")
    lines += ["", "## Limits and next decision", "",
        "This finishes the predeclared candidate comparison. Do not continue tuning on these validation outcomes in this round.",
        "The original predict.py and baseline bundle remain unchanged. Any eventual promotion must explicitly identify its operating policy and limitations.",
        "A final reliability claim needs newly preselected independent tracks/artists, fixed model/thresholds, an overlap audit, and uncertainty estimates. The 90 old test tracks cannot be relabeled as unseen.",
        "The small coverage-enriched sample, incomplete uploader genre labels, only four target genres, and first-30-second cropping limit generalization to arbitrary uploads.",
        "Weight license: CC BY-NC 4.0 as identified by the model card. Final distribution review for our code, the pretrained weights, classifier and example audio remains separate and incomplete.", "",
        "## Reproduce locally", "", "```bash", "python3 -m venv .venv-improve",
        ".venv-improve/bin/python -m pip install -r requirements-mert-lock.txt",
        ".venv-improve/bin/python prepare_mert.py", ".venv-improve/bin/python extract_mert.py pilot --device cpu",
        ".venv-improve/bin/python extract_mert.py extract --device cpu",
        ".venv-improve/bin/python improve_model.py mert --cache outputs/improvement/mert_features.npz",
        ".venv-improve/bin/python report_mert.py", "```", "",
        "Requires the prepared 480-track project data and original baseline artifacts. No corpus audio is downloaded by the encoder setup; only public model files are retrieved.",
        "If necessary, the downloader accepts --proxy with your existing proxy URL; it does not change system network settings. Do not disable TLS verification.",
        "The upstream optional nnAudio warning is expected: CQT is disabled in this pinned checkpoint, so no nnAudio install is needed.", "",
        "Sources: [pinned MERT model card](https://huggingface.co/m-a-p/MERT-v0-public/tree/e8413e398b3180ca488534aea68fd829402044a4), [PyTorch restricted checkpoint loading](https://docs.pytorch.org/docs/2.8/generated/torch.load.html).", ""]
    (ROOT / "MERT_RESULTS.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), layout="constrained")
    pos = np.arange(4)
    for ax, metric in zip(axes, ("precision", "recall"), strict=True):
        for off, report, label, color in [(-.18, a, "MFCC + selected head", "#747985"), (.18, b, "MERT + selected head", "#146e87")]:
            bars = ax.bar(pos + off, [r[metric] for r in report["per_label"]], .34, color=color, label=label)
            ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
        ax.set(xticks=pos, xticklabels=results["mert"]["labels"], ylim=(0, 1.15), ylabel=metric.title(), title=f"Validation {metric} (F1-oriented policy)")
        ax.legend(frameon=False, fontsize=8, loc="upper center")
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Same 90 development tracks; not an independent final test", fontsize=13)
    fig.savefig(OUT / "mert_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
