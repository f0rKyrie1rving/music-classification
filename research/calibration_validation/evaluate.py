"""Apply the saved primary classifier and calibrators without any fitting.

The verify-model stage checks only historical arrays. New validation predictions
are generated solely by the score stage, after the acquisition and feature cache
have been completed and the model/protocol references have been frozen.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.special import expit

from research.calibration.common import LABELS, ROOT, development, digest, now, read, save, targets
from research.calibration.metrics import evaluate, paired_bootstrap
from research.calibration_conservative.run import verify as verify_conservative, verify_selection

METHODS = ("identity", "sigmoid", "conservative_sigmoid")
FEATURE_DIM = 2304
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 2026092702


def relative_file(root, name):
    """Reject ambiguous references while keeping frozen paths relocatable."""
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a relative frozen path: {name}")
    return root / path


def logits_from_parameters(features, parameters):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != FEATURE_DIM or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected finite nonempty N x 2304 features")
    heads = parameters["heads"]
    if set(heads) != set(LABELS):
        raise ValueError("Saved classifier label set changed")
    columns = []
    for label in LABELS:
        head = heads[label]
        mean = np.asarray(head["scaler_mean"], dtype=np.float64)
        scale = np.asarray(head["scaler_scale"], dtype=np.float64)
        coef = np.asarray(head["coef"], dtype=np.float64)
        intercept = np.asarray(head["intercept"], dtype=np.float64)
        if (mean.shape != (FEATURE_DIM,) or scale.shape != (FEATURE_DIM,)
                or coef.shape != (1, FEATURE_DIM) or intercept.shape != (1,)
                or head["classes"] != [0, 1]):
            raise ValueError(f"Unexpected saved parameter shape/classes: {label}")
        if (not all(np.isfinite(a).all() for a in (mean, scale, coef, intercept))
                or np.any(scale <= 0)):
            raise ValueError(f"Invalid saved classifier values: {label}")
        # Match StandardScaler.transform then LogisticRegression.decision_function.
        transformed = x.copy()
        transformed -= mean
        transformed /= scale
        columns.append((transformed @ coef.T + intercept).ravel())
    z = np.column_stack(columns)
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite reconstructed logits")
    return z


def probabilities_from_parameters(logits, parameters, policy):
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != len(LABELS) or not len(z) or not np.isfinite(z).all():
        raise ValueError("Expected finite nonempty N x 4 logits")
    if set(policy["labels"]) != set(LABELS):
        raise ValueError("Saved conservative policy label set changed")
    raw = expit(z)
    full = np.empty_like(raw)
    conservative = np.empty_like(raw)
    for j, label in enumerate(LABELS):
        fitted = parameters["calibrators"]["sigmoid"][label]
        a, b = float(fitted["slope"]), float(fitted["intercept"])
        if not fitted["success"] or not np.isfinite([a, b]).all() or a <= 0:
            raise ValueError(f"Unusable saved sigmoid: {label}")
        full[:, j] = expit(a * z[:, j] + b)
        item = policy["labels"][label]
        strength = float(item["selected_lambda"])
        if strength not in (0., .25, .5, .75, 1.):
            raise ValueError(f"Unexpected saved correction strength: {label}")
        policy_fit = item["full_fit"]
        if policy_fit is not None:
            if (not policy_fit["success"] or policy_fit["slope"] != a
                    or policy_fit["intercept"] != b):
                raise ValueError(f"Conservative/full sigmoid parameter mismatch: {label}")
        elif strength != 0:
            raise ValueError(f"Nonzero correction without saved full fit: {label}")
        conservative[:, j] = (1 - strength) * raw[:, j] + strength * full[:, j]
    result = dict(identity=raw, sigmoid=full, conservative_sigmoid=conservative)
    if any(not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) for p in result.values()):
        raise ValueError("Invalid predicted probabilities")
    return result


def load_model(out):
    reference = read(out / "model_reference.json")
    base = relative_file(ROOT, reference["base_run"])
    conservative = relative_file(ROOT, reference["conservative_run"])
    seed = reference["primary_seed"]
    config, _, _, splits = verify_conservative(conservative)
    verify_selection(conservative)
    if (seed != config["primary_seed"] or reference["base_run"] != config["base_run"]
            or seed != 20260926):
        raise ValueError("Validation must use the originally designated primary model")
    split = next(s for s in splits if s["seed"] == seed)
    parameters = read(base / f"seed_{seed}" / "parameters.json")
    policy = read(conservative / "selection" / f"{seed}_policy.json")
    if (parameters["classifier_fit_ids"] != split["subsets"]["fit"]["ids"]
            or parameters["calibrator_fit_ids"] != split["subsets"]["calibration"]["ids"]):
        raise ValueError("Saved model fitting IDs changed")
    return reference, base, conservative, split, parameters, policy


def require_absent(out, names):
    for name in names:
        if (out / name).exists():
            raise FileExistsError(out / name)


def verify_model(out):
    require_absent(out, ("numerical_checks.json", "model_hashes.json"))
    reference, base, conservative, split, parameters, policy = load_model(out)
    rows, y, groups = development()
    features = np.load(ROOT / "outputs/maest_hf/features.npy", allow_pickle=False)
    metadata = read(ROOT / "outputs/maest_hf/features.json")
    if metadata["ids"] != [r["track_id"] for r in rows] or features.shape != (1206, FEATURE_DIM):
        raise ValueError("Historical feature cache alignment changed")
    checks = {}
    prior_path = base / f"seed_{reference['primary_seed']}" / "predictions.npz"
    with np.load(prior_path, allow_pickle=False) as prior:
        for role in ("calibration", "evaluation"):
            ix = split["indices"][role]
            z = logits_from_parameters(features[ix], parameters)
            ids = [rows[i]["track_id"] for i in ix]
            if ids != prior[f"{role}_ids"].tolist() or not np.array_equal(y[ix], prior[f"{role}_y"]):
                raise ValueError(f"Historical {role} ID/target mismatch")
            np.testing.assert_allclose(z, prior[f"{role}_logits"], atol=1e-12, rtol=0)
            checks[role] = {"tracks": len(ix), "max_abs_logit_error": float(np.max(np.abs(z - prior[f"{role}_logits"])))}
            if role == "evaluation":
                probabilities = probabilities_from_parameters(z, parameters, policy)
                for method in ("identity", "sigmoid"):
                    np.testing.assert_allclose(probabilities[method], prior[method], atol=1e-12, rtol=0)
                    checks[role][f"max_abs_{method}_error"] = float(np.max(np.abs(probabilities[method] - prior[method])))
                if not np.array_equal(groups[ix], prior["evaluation_artists"]):
                    raise ValueError("Historical evaluation artist order mismatch")
    saved_conservative = conservative / "evaluation" / f"{reference['primary_seed']}_predictions.npz"
    evaluation_status = read(conservative / "evaluation/status.json")
    if (evaluation_status["state"] != "complete"
            or digest(saved_conservative) != evaluation_status["file_hashes"][saved_conservative.name]):
        raise ValueError("Historical conservative evaluation archive changed")
    with np.load(saved_conservative, allow_pickle=False) as prior:
        if prior["evaluation_ids"].tolist() != ids:
            raise ValueError("Historical conservative ID order mismatch")
        np.testing.assert_allclose(probabilities["conservative_sigmoid"], prior["conservative_sigmoid"], atol=1e-12, rtol=0)
        checks["evaluation"]["max_abs_conservative_error"] = float(np.max(np.abs(probabilities["conservative_sigmoid"] - prior["conservative_sigmoid"])))
    files = [base / "freeze.json", base / "config.json", base / "splits.json",
             base / f"seed_{reference['primary_seed']}" / "parameters.json", prior_path,
             conservative / "freeze.json", conservative / "config.json",
             conservative / "selection/status.json",
             conservative / "selection" / f"{reference['primary_seed']}_policy.json",
             conservative / "evaluation/status.json", saved_conservative,
             ROOT / "outputs/maest_hf/features.npy", ROOT / "outputs/maest_hf/features.json",
             ROOT / "data/expanded_manifest.json", ROOT / "experiments/final_holdout_plan.json",
             ROOT / "expanded_model.py"]
    model_hashes = {"created_utc": now(), "reference_sha256": digest(out / "model_reference.json"),
                    "files": {str(p.relative_to(ROOT)): digest(p) for p in files}}
    save(out / "model_hashes.json", model_hashes)
    save(out / "numerical_checks.json", {"state": "passed", "checked_utc": now(),
         "absolute_tolerance": 1e-12, "refitted": False, "checks": checks,
         "model_hashes_sha256": digest(out / "model_hashes.json")})
    print("Saved primary model reconstructed successfully:", checks, flush=True)


def verify_frozen(out):
    frozen = read(out / "freeze.json")
    for key in ("source_hashes", "input_hashes"):
        for name, expected in frozen[key].items():
            if digest(relative_file(ROOT, name)) != expected:
                raise ValueError(f"Changed frozen file: {name}")
    for name, expected in frozen["local_hashes"].items():
        if digest(relative_file(out, name)) != expected:
            raise ValueError(f"Changed frozen local file: {name}")
    for name in ("model_reference.json", "model_hashes.json", "numerical_checks.json"):
        if name not in frozen["local_hashes"]:
            raise ValueError(f"Missing frozen model reference: {name}")
    numerical = read(out / "numerical_checks.json")
    model_hashes = read(out / "model_hashes.json")
    if (numerical["state"] != "passed" or numerical["refitted"] is not False
            or numerical["model_hashes_sha256"] != digest(out / "model_hashes.json")
            or model_hashes["reference_sha256"] != digest(out / "model_reference.json")):
        raise ValueError("Model reconstruction receipt changed")
    for name, expected in model_hashes["files"].items():
        if digest(relative_file(ROOT, name)) != expected:
            raise ValueError(f"Changed saved model input: {name}")
    return frozen


def validate_feature_order(rows, features, metadata):
    ids = [row["track_id"] for row in rows]
    artists = [row["artist_id"] for row in rows]
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids + artists):
        raise ValueError("Empty, duplicate, or invalid validation IDs/artists")
    if metadata["ids"] != ids:
        raise ValueError("Validation feature ID order differs from manifest")
    if (features.shape != (len(ids), FEATURE_DIM) or features.dtype != np.float32
            or not np.isfinite(features).all()):
        raise ValueError("Expected finite N x 2304 float32 validation features")
    if any(not isinstance(row["tags"], list) or any(not isinstance(t, str) for t in row["tags"]) for row in rows):
        raise ValueError("Invalid validation source tags")
    return np.asarray(ids), np.asarray(artists)


def validate_observation_partition(selected, observed, acquisition, extraction):
    """Every fixed selected row must survive or have one recorded failure."""
    def unique(records, description):
        identifiers = [r["track_id"] for r in records]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"Duplicate {description} IDs")
        return set(identifiers)

    chosen = unique(selected, "selected")
    retained = unique(observed, "observed")
    downloaded = unique(acquisition["successful"], "download success")
    download_failed = unique(acquisition["failures"], "download failure")
    extraction_failed = unique(extraction["extraction_failures"], "extraction failure")
    if (extraction["state"] != "complete" or acquisition["requested"] != len(selected)
            or acquisition["replacement_tracks"] != 0
            or downloaded & download_failed or downloaded | download_failed != chosen
            or retained & extraction_failed or retained | extraction_failed != downloaded
            or extraction["download_failures"] != acquisition["failures"]):
        raise ValueError("Observed/download/extraction failure partition does not reconcile")
    expected_rows = [r for r in selected if r["track_id"] in retained]
    if observed != expected_rows:
        raise ValueError("Observed metadata/order changed from selected sample")
    if (extraction["selected_tracks"] != len(selected)
            or extraction["selected_artists"] != len({r["artist_id"] for r in selected})
            or extraction["observed_tracks"] != len(observed)
            or extraction["observed_artists"] != len({r["artist_id"] for r in observed})):
        raise ValueError("Observed coverage counts do not reconcile")


def verify_observation_artifacts(out, rows, metadata):
    """Verify acquisition and extraction accounting before exposing test scores."""
    selected = read(out / "selected_manifest.json")["tracks"]
    acquisition = read(out / "download_status.json")
    extraction = read(out / "extraction_status.json")
    validate_observation_partition(selected, rows, acquisition, extraction)
    for key, name in (("selected_manifest_sha256", "selected_manifest.json"),
                      ("observed_manifest_sha256", "manifest.json"),
                      ("download_status_sha256", "download_status.json"),
                      ("runtime_control_sha256", "runtime_control.json")):
        if extraction[key] != digest(out / name):
            raise ValueError(f"Extraction provenance mismatch: {name}")
    runtime = read(out / "runtime_control.json")
    error, tolerance = runtime["max_absolute_feature_error"], runtime["absolute_tolerance"]
    if (not np.isfinite([error, tolerance]).all() or error < 0 or tolerance != 1e-5
            or error > tolerance or runtime["device"] != "cpu"
            or metadata["device"] != "cpu" or runtime["versions"] != metadata["versions"]):
        raise ValueError("Historical encoder control or validation runtime is invalid")
    if (metadata["shape"] != [len(rows), FEATURE_DIM] or metadata["dtype"] != "float32"
            or metadata["extractor_sha256"] != digest(ROOT / "maest_hf_features.py")
            or metadata["encoder_receipt_sha256"] != digest(ROOT / "models/mtg-upf-maest-519l/SOURCES.json")):
        raise ValueError("Feature extraction provenance changed")
    successes = {r["track_id"]: r for r in acquisition["successful"]}
    audio_hashes = []
    paths = [out / name for name in ("selected_manifest.json", "download_status.json",
                                   "extraction_status.json", "runtime_control.json")]
    for row in rows:
        receipt_path = out / "acquisition/download_receipts" / f"{row['track_id']}.json"
        receipt = read(receipt_path)
        audio_path = relative_file(ROOT, row["audio_file"])
        if (receipt != successes[row["track_id"]] or receipt["track_id"] != row["track_id"]
                or digest(audio_path) != receipt["wav_sha256"]):
            raise ValueError(f"Validation audio/receipt changed: {row['track_id']}")
        audio_hashes.append(receipt["wav_sha256"])
        paths.extend((receipt_path, audio_path))
    if metadata["audio_hashes"] != audio_hashes:
        raise ValueError("Feature metadata audio order/hashes do not match observed rows")
    y = targets(rows)
    artists = np.asarray([r["artist_id"] for r in rows])
    if (extraction["positive_tracks"] != y.sum(0).tolist()
            or extraction["positive_artists"] != [len(set(artists[y[:, j] == 1])) for j in range(4)]):
        raise ValueError("Extraction label coverage counts changed")
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def score(out):
    names = ("predictions.npz", "metrics.json", "summary.json", "intervals.json",
             "raw_draws.npz", "full_sigmoid_draws.npz", "score_status.json")
    require_absent(out, names)
    verify_frozen(out)
    reference, _, _, _, parameters, policy = load_model(out)
    rows = read(out / "manifest.json")["tracks"]
    metadata = read(out / "features.json")
    if digest(out / "features.npy") != metadata["feature_sha256"]:
        raise ValueError("Validation feature file hash changed")
    features = np.load(out / "features.npy", allow_pickle=False)
    ids, artists = validate_feature_order(rows, features, metadata)
    acquisition_hashes = verify_observation_artifacts(out, rows, metadata)
    if len(set(artists)) < 2:
        raise ValueError("At least two validation artists are needed for grouped intervals")
    historical = read(ROOT / "data/expanded_manifest.json")["tracks"]
    if (set(ids) & {r["track_id"] for r in historical}
            or set(artists) & {r["artist_id"] for r in historical}):
        raise ValueError("Validation tracks/artists overlap the historical project pool")
    input_hashes = {name: digest(out / name) for name in
                    ("freeze.json", "manifest.json", "features.json", "features.npy",
                     "model_reference.json", "model_hashes.json", "numerical_checks.json")}
    save(out / "score_status.json", {"state": "running", "started_utc": now(),
         "input_hashes": input_hashes, "acquisition_hashes": acquisition_hashes})
    try:
        y = targets(rows)
        logits = logits_from_parameters(features, parameters)
        probabilities = probabilities_from_parameters(logits, parameters, policy)
        metrics = {method: evaluate(y, p) for method, p in probabilities.items()}
        np.savez_compressed(out / "predictions.npz", ids=ids, artists=artists, y=y, logits=logits, **probabilities)
        save(out / "metrics.json", metrics)
        intervals = {}
        for name, baseline in (("raw", "identity"), ("full_sigmoid", "sigmoid")):
            comparison = {"identity": probabilities[baseline], "conservative_sigmoid": probabilities["conservative_sigmoid"]}
            result, draws = paired_bootstrap(y, comparison, artists, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
            result["baseline"] = baseline
            result["direction"] = f"conservative sigmoid minus {baseline}; negative favors conservative sigmoid"
            result["scope"] = ("Pointwise artist-cluster percentile intervals, conditional on the fixed classifier, "
                               "calibration data and selected policy. No multiplicity adjustment or refitting uncertainty.")
            intervals[name] = result
            np.savez_compressed(out / f"{name}_draws.npz", **draws["conservative_sigmoid"])
        save(out / "intervals.json", intervals)
        summary = {"primary_seed": reference["primary_seed"], "tracks": len(ids), "artists": len(set(artists)),
                   "primary_metric": "track-weighted equally label-weighted macro binary Brier",
                   "primary_comparison": "conservative_sigmoid minus identity",
                   "selected_strengths": {label: policy["labels"][label]["selected_lambda"] for label in LABELS},
                   "methods": [{"method": m, "macro_brier": metrics[m]["macro_brier"],
                      "macro_log_loss": metrics[m]["macro_log_loss"],
                      "delta_brier_vs_raw": metrics[m]["macro_brier"] - metrics["identity"]["macro_brier"],
                      "delta_brier_vs_full_sigmoid": metrics[m]["macro_brier"] - metrics["sigmoid"]["macro_brier"]}
                     for m in METHODS],
                   "primary_interval": intervals["raw"]["results"]["conservative_sigmoid"]["brier"]["macro"],
                   "secondary_interval": intervals["full_sigmoid"]["results"]["conservative_sigmoid"]["brier"]["macro"]}
        save(out / "summary.json", summary)
        # A concurrent acquisition/cache change invalidates this evaluation.
        for name, expected in input_hashes.items():
            if digest(out / name) != expected:
                raise ValueError(f"Scoring input changed during evaluation: {name}")
        for name, expected in acquisition_hashes.items():
            if digest(ROOT / name) != expected:
                raise ValueError(f"Acquisition input changed during evaluation: {name}")
        verify_frozen(out)
        save(out / "score_status.json", {"state": "complete", "completed_utc": now(),
             "input_hashes": input_hashes, "acquisition_hashes": acquisition_hashes,
             "output_hashes": {name: digest(out / name) for name in names if name != "score_status.json"},
             "classifier_or_calibrator_refitted": False})
        print("Independent validation completed:", summary, flush=True)
    except Exception as error:
        save(out / "score_status.json", {"state": "failed", "failed_utc": now(), "error": repr(error),
             "input_hashes": input_hashes, "acquisition_hashes": acquisition_hashes})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify-model", "score"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    {"verify-model": verify_model, "score": score}[args.action](args.out.resolve())
