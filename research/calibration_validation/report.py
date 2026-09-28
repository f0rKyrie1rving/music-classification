"""Render a neutral numerical report after fixed-model validation is complete."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from research.calibration.common import LABELS, ROOT, digest, read, targets
from research.calibration.report import plots
from .evaluate import METHODS, verify_frozen


def interval_row(reference, metric, record):
    lo, hi = record["ci95"]
    return f"| {reference} | {metric} | {record['delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |"


def create(out, destination):
    if destination.exists():
        raise FileExistsError(destination)
    verify_frozen(out)
    status = read(out / "score_status.json")
    if status["state"] != "complete" or status["classifier_or_calibrator_refitted"] is not False:
        raise ValueError("Report requires a completed fixed-model evaluation")
    for name, expected in status["input_hashes"].items():
        if digest(out / name) != expected:
            raise ValueError(f"Evaluation input changed: {name}")
    for name, expected in status["output_hashes"].items():
        if digest(out / name) != expected:
            raise ValueError(f"Evaluation output changed: {name}")
    for name, expected in status["acquisition_hashes"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Acquisition artifact changed: {name}")
    verification = read(out / "verification.json")
    if verification.get("status") != "passed" or verification.get("errors") != []:
        raise ValueError("Independent numerical verification has not passed")
    for name, expected in verification["verified_artifact_hashes"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Independently verified artifact changed: {name}")
    summary = read(out / "summary.json")
    metrics = read(out / "metrics.json")
    intervals = read(out / "intervals.json")
    extraction = read(out / "extraction_status.json")
    selected = read(out / "selected_manifest.json")["tracks"]
    observed = read(out / "manifest.json")["tracks"]
    audit = read(out / "data_audit.json")
    config = read(out / "config.json")
    exposure = read(out / "exposure_history.json")
    with np.load(out / "predictions.npz", allow_pickle=False) as data:
        y = data["y"].copy()
        artists = data["artists"].copy()
        probabilities = {method: data[method].copy() for method in METHODS}
        if data["ids"].tolist() != [r["track_id"] for r in observed]:
            raise ValueError("Report prediction/manifest order mismatch")
    if not np.array_equal(y, targets(observed)):
        raise ValueError("Report target alignment changed")
    figures = destination / "figures"
    results = destination / "results"
    figures.mkdir(parents=True)
    results.mkdir()
    names = ["predictions.npz", "metrics.json", "summary.json", "intervals.json",
             "raw_draws.npz", "full_sigmoid_draws.npz", "score_status.json",
             "verification.json", "config.json", "freeze.json", "model_reference.json",
             "model_hashes.json", "numerical_checks.json", "data_audit.json",
             "selected_manifest.json", "manifest.json", "download_status.json",
             "extraction_status.json", "features.json", "runtime_control.json",
             "exposure_history.json", "exclusions.json"]
    for name in names:
        shutil.copy2(out / name, results / name)
    shutil.copy2(out / "protocol.md", destination / "PROTOCOL.md")
    plots(figures, y, probabilities, "New project-ID validation: fixed primary classifier and probability mappings")

    table = ["| Method | Macro Brier | Difference versus raw | Difference versus full sigmoid | Macro log loss |",
             "| --- | --- | --- | --- | --- |"]
    for method in METHODS:
        value = metrics[method]
        table.append(f"| {method} | {value['macro_brier']:.6f} | "
                     f"{value['macro_brier'] - metrics['identity']['macro_brier']:+.6f} | "
                     f"{value['macro_brier'] - metrics['sigmoid']['macro_brier']:+.6f} | {value['macro_log_loss']:.6f} |")
    ci_table = ["| Comparator | Metric | Conservative difference | Conditional 95% interval |",
                "| --- | --- | --- | --- |"]
    for reference, comparator in (("raw", "raw probabilities"), ("full_sigmoid", "full sigmoid")):
        for metric in ("brier", "log_loss"):
            ci_table.append(interval_row(comparator, metric,
                intervals[reference]["results"]["conservative_sigmoid"][metric]["macro"]))
    label_table = ["| Label | Raw Brier | Full sigmoid Brier | Conservative Brier | Conservative − raw |",
                   "| --- | --- | --- | --- | --- |"]
    for label in LABELS:
        values = [metrics[m]["per_label"][label]["brier"] for m in METHODS]
        label_table.append(f"| {label} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | {values[2]-values[0]:+.6f} |")
    selected_y = targets(selected)
    selected_artists = np.asarray([r["artist_id"] for r in selected])
    coverage_table = ["| Quantity | Fixed selected sample | Observed/scored sample |",
                      "| --- | --- | --- |",
                      f"| Tracks | {len(selected)} | {len(observed)} |",
                      f"| Artists | {len(set(selected_artists))} | {len(set(artists))} |"]
    for j, label in enumerate(LABELS):
        coverage_table.append(f"| {label}: positive tracks | {int(selected_y[:,j].sum())} | {int(y[:,j].sum())} |")
        coverage_table.append(f"| {label}: positive artists | {len(set(selected_artists[selected_y[:,j]==1]))} | {len(set(artists[y[:,j]==1]))} |")
    coverage_table.append(f"| No positive broad label | {int((selected_y.sum(1)==0).sum())} | {int((y.sum(1)==0).sum())} |")
    coverage_table.append(f"| Multiple positive broad labels | {int((selected_y.sum(1)>1).sum())} | {int((y.sum(1)>1).sum())} |")
    strengths = ", ".join(f"{label} {summary['selected_strengths'][label]:.0%}" for label in LABELS)
    track_coverage = len(observed) / len(selected)
    artist_coverage = len(set(artists)) / len(set(selected_artists))
    run_name = out.name
    text = f"""# Fixed probability-calibration validation on new documented project IDs

Run `{run_name}` evaluates the previously designated primary research classifier and
its already selected probability correction. This is a new sample relative to documented
project track and artist IDs. It does not establish freedom from undocumented listening,
artist aliases/collaborations, or upstream encoder-pretraining exposure.

## Fixed comparison and numerical outcomes

The classifier was fitted on 733 historical development songs; the sigmoid mappings
were fitted on 232 different songs. The saved conservative strengths are {strengths}.
The encoder, classifier, calibration parameters and strengths were not refitted.
The same songs are scored by raw, full sigmoid and conservative sigmoid methods.

The primary endpoint is **conservative minus raw track-weighted macro binary Brier**,
with equal weight for the four labels. Lower Brier/log loss is better; a negative
paired difference favors conservative calibration. Full sigmoid is a fixed secondary
comparator. Tables report all outcomes without selecting the favorable comparison.

{chr(10).join(table)}

{chr(10).join(ci_table)}

Intervals use {config['bootstrap_replicates']:,} paired artist-cluster resamples with
seed {config['bootstrap_seed']}; both comparisons reuse the same sampled artists.
They are conditional on the fitted classifier, calibration data and selected policy.
They omit refitting/selection uncertainty and multiplicity adjustment. A 95% interval
crossing zero is inconclusive about the mean change, not evidence of equivalence.
The archive retains full precision; displayed values are rounded to six decimals.

{chr(10).join(label_table)}

## Sample and observed coverage

The metadata-only candidate frame contained {audit['counts']['eligible_tracks']} tracks
from {audit['counts']['eligible_artists']} artists in archive shards
{', '.join(config['archives'])}. A fixed hash ordering selected {config['artist_count']}
artists and up to {config['tracks_per_artist']} eligible tracks per artist. Selection
used neither model scores nor genre quotas. All-zero and multi-positive broad targets
were retained. The source records have at least one genre annotation; the four broad
targets are the pre-existing ontology applied to uploader tags.

{chr(10).join(coverage_table)}

Track coverage was {track_coverage:.2%}; artist coverage was {artist_coverage:.2%}.
There were {len(extraction['download_failures'])} recorded acquisition failures and
{len(extraction['extraction_failures'])} recorded extraction failures. Failed rows were
not replaced. The estimated contrast concerns the observed sample; missing audio or
failed extraction can bias it. Complete failure details are in
[extraction status](results/extraction_status.json), and all selected IDs remain in
the [selected manifest](results/selected_manifest.json).

The bounded four-shard frame, licensing restrictions, genre-annotation requirement,
exclusion of previously used artists and artist-first sampling limit generalization.
Two songs from an artist do not count as independent artist observations. The primary
track-weighted score gives two-song artists twice the weight of one-song artists.
The fixed sample size is a practical first validation, not a power guarantee; no extra
songs were acquired in response to a confidence interval or label count.

## Probability quality and reliability

![Five-bin reliability curves](figures/reliability_5.png)

![Ten-bin sensitivity curves](figures/reliability_10.png)

Curves use fixed equal-width bins; lower panels show bin counts. Sparse and empty bins
must be interpreted with their counts, and the curves have no pointwise uncertainty
bands. Brier and log loss measure overall probability quality, not pure calibration
or music-recognition accuracy in isolation. These positive monotone labelwise mappings
preserve ranking apart from numerical ties. Each label remains an independent binary
output; probabilities are not normalized to sum to one.

## Independence, provenance and use limits

Candidate metadata and acquisition availability were inspected before predictions.
The local protocol was frozen before new audio predictions; it was not externally
preregistered. All 1,535 historical project tracks and their 583 artist IDs were
excluded. Newness is conditional on those documented records. The external-exposure
record is reproduced below exactly as structured data; an unanswered question is
unconfirmed history, not a declaration of no prior exposure.

```json
{json.dumps(exposure, ensure_ascii=False, indent=2)}
```

Source tags are incomplete uploader annotations for whole tracks, while the frozen
preprocessing uses the first 30 seconds. Artist IDs cannot rule out aliases, shared
performers or duplicate recordings. Encoder-pretraining track overlap remains unknown.
This is validation within a restricted MTG-Jamendo frame, not an external-catalogue
or streaming-population accuracy claim.

Saved classifier reconstruction and a historical encoder control were checked before
new prediction. Acquisition receipts retain byte ranges, local WAV hashes and any
download failures. Partial MP3 prefixes are not described as verified complete files.
See [model numerical checks](results/numerical_checks.json), [encoder control](results/runtime_control.json),
[independent verification](results/verification.json), [frozen protocol](PROTOCOL.md),
and [full predictions](results/predictions.npz). This report package contains numerical
artifacts and metadata, not source audio or model weights.

The research classifier differs from the application's final classifier fitted on all
1,206 development songs. This experiment does not change the application or justify
copying its calibration parameters onto that different classifier. Once these outcomes
have been inspected, these songs are no longer an untouched validation set for later
method revisions. Subsequent development must disclose this use and require another
separately fixed sample for fresh confirmation.

The source is the [official MTG-Jamendo dataset](https://github.com/MTG/mtg-jamendo-dataset)
at Git commit `{config['source_commit']}`. The retained metadata, checksums and individual
license records define the acquisition frame. See the [data audit](results/data_audit.json)
and [source/model hashes](results/freeze.json). Audio was used locally for non-commercial
research under the retained source conditions.
"""
    (destination / "REPORT.md").write_text(text)
    print("Wrote fixed-model validation report:", destination / "REPORT.md", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    create(args.out.resolve(), args.destination.resolve())
