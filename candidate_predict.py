"""Run the fixed experimental candidate, separately from the installed application."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import predict_app
from maest_hf_features import FEATURE_WIDTH, HIDDEN_STATE_INDEX, HfMaestEncoder
from prepare_maest_hf import MODEL_ID, REVISION
from predict_maest import LABELS, POLICIES, REPRESENTATION


ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "artifacts/candidates/20260927_v1"
REQUIRED_KEYS = {
    "format_version", "candidate_id", "status", "labels", "representation",
    "feature_width", "encoder", "weights_file", "weights_sha256", "thresholds",
    "precision_supported", "head_provenance", "baseline_reference", "training_file",
    "training_sha256", "selection_file", "selection_sha256", "freeze_file",
    "freeze_sha256", "future_exclusions_file", "future_exclusions_sha256", "license",
}
ARRAY_SHAPES = {
    "means": (4, FEATURE_WIDTH), "scales": (4, FEATURE_WIDTH),
    "coefficients": (4, FEATURE_WIDTH), "intercepts": (4,), "offsets": (4,),
}
REFERENCES = {
    "weights": "candidate.npz", "training": "training.json",
    "selection": "threshold_selection.json", "freeze": "freeze.json",
    "future_exclusions": "future_exclusions.json",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _valid_hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _validate_parameters(bundle):
    for name, shape in ARRAY_SHAPES.items():
        value = bundle.get(name)
        if (not isinstance(value, np.ndarray) or value.dtype != np.float64
                or value.shape != shape or not np.isfinite(value).all()):
            raise ValueError(f"Invalid candidate parameter array: {name}.")
    if np.any(bundle["scales"] <= 0) or np.any(bundle["offsets"] != 0):
        raise ValueError("Candidate requires positive scales and zero logit offsets.")


def _validate_thresholds(bundle):
    thresholds = bundle.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != set(POLICIES):
        raise ValueError("Candidate threshold policies changed.")
    for policy in POLICIES:
        try:
            values = np.asarray(thresholds[policy], dtype=np.float64)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid candidate thresholds: {policy}.") from None
        limit = 1.0 if policy == "f1" else 1.01
        if (values.shape != (4,) or not np.isfinite(values).all()
                or np.any(values < 0) or np.any(values > limit)):
            raise ValueError(f"Invalid candidate thresholds: {policy}.")
    if thresholds["precision_target"] != [0.625, 1.01, 1.01, 0.525]:
        raise ValueError("Candidate selective policy must preserve the installed policy.")
    if (thresholds["f1"][0] != 0.39999999999999997
            or thresholds["f1"][3] != 0.27499999999999997):
        raise ValueError("Copied electronic/rock thresholds changed.")


def load_candidate(directory=DEFAULT):
    """Validate a portable, non-executable candidate and all five bundled receipts."""
    directory = Path(directory).resolve()
    bundle = json.loads((directory / "candidate.json").read_text(encoding="utf-8"))
    if (not isinstance(bundle, dict) or set(bundle) != REQUIRED_KEYS
            or type(bundle["format_version"]) is not int or bundle["format_version"] != 1):
        raise ValueError("Candidate metadata has unexpected fields or format.")
    if (bundle["candidate_id"] != "music-candidate-20260927-v1"
            or bundle["status"] != "candidate_not_fresh_validated"
            or bundle["labels"] != list(LABELS)
            or bundle["representation"] != REPRESENTATION
            or type(bundle["feature_width"]) is not int
            or bundle["feature_width"] != FEATURE_WIDTH
            or bundle["encoder"] != {"model_id": MODEL_ID, "revision": REVISION,
                                     "hidden_state_index": HIDDEN_STATE_INDEX}
            or bundle["precision_supported"] != [True, False, False, True]):
        raise ValueError("Candidate identity or encoder metadata changed.")
    if not isinstance(bundle["license"], str) or not bundle["license"]:
        raise ValueError("Candidate license is missing.")
    provenance = bundle["head_provenance"]
    if (not isinstance(provenance, list) or len(provenance) != 4
            or any(not isinstance(item, dict) or item.get("label") != label
                   or not isinstance(item.get("kind"), str) or not item["kind"]
                   for item, label in zip(provenance, LABELS))):
        raise ValueError("Invalid candidate head provenance.")
    baseline = bundle["baseline_reference"]
    if (not isinstance(baseline, dict) or not baseline
            or any(not isinstance(path, str) or not path or not _valid_hash(digest)
                   for path, digest in baseline.items())):
        raise ValueError("Invalid candidate baseline reference.")
    _validate_thresholds(bundle)
    for prefix, filename in REFERENCES.items():
        name, digest = bundle[f"{prefix}_file"], bundle[f"{prefix}_sha256"]
        if name != filename or not _valid_hash(digest):
            raise ValueError(f"Invalid candidate file reference: {prefix}.")
        path = (directory / name).resolve()
        if path.parent != directory or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Candidate file missing or checksum mismatch: {name}.")
        if prefix != "weights":
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError(f"Candidate receipt must be a JSON object: {name}.")
    with np.load(directory / bundle["weights_file"], allow_pickle=False) as archive:
        if set(archive.files) != set(ARRAY_SHAPES) or len(archive.files) != len(ARRAY_SHAPES):
            raise ValueError("Candidate parameter archive has unexpected arrays.")
        bundle.update({name: archive[name].copy() for name in ARRAY_SHAPES})
    _validate_parameters(bundle)
    return bundle


def predict_arrays(bundle, features, policy="f1"):
    """Return full precision arrays using the installed application's row arithmetic."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown prediction policy: {policy}.")
    _validate_parameters(bundle)
    features = np.asarray(features, dtype=np.float64)
    if (features.ndim != 2 or features.shape[1] != FEATURE_WIDTH
            or not np.isfinite(features).all()):
        raise ValueError(f"Expected finite MAEST feature rows of width {FEATURE_WIDTH}.")
    thresholds = np.asarray(bundle["thresholds"][policy], dtype=np.float64)
    if thresholds.shape != (4,) or not np.isfinite(thresholds).all():
        raise ValueError("Invalid prediction thresholds.")
    logits = np.empty((len(features), 4), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for index, feature in enumerate(features):
            logits[index] = np.sum(
                ((feature - bundle["means"]) / bundle["scales"])
                * bundle["coefficients"], axis=1) + bundle["intercepts"]
    if not np.isfinite(logits).all():
        raise ValueError("Candidate classifier produced non-finite logits.")
    raw = np.exp(-np.logaddexp(0.0, -logits))
    probability = raw.copy()  # Every head in this fixed candidate is unweighted.
    return {"logits": logits, "raw": raw, "probability": probability,
            "decision": probability >= thresholds}


def predict_vector(bundle, feature, policy="f1"):
    feature = np.asarray(feature, dtype=np.float64)
    if feature.shape != (FEATURE_WIDTH,) or not np.isfinite(feature).all():
        raise ValueError(f"Expected one finite MAEST feature vector of width {FEATURE_WIDTH}.")
    values = predict_arrays(bundle, feature[None, :], policy)
    thresholds = bundle["thresholds"][policy]
    return [{"label": label, "score": round(float(values["probability"][0, index]), 4),
             "threshold": round(float(thresholds[index]), 4),
             "selected": bool(values["decision"][0, index]),
             "raw_score": round(float(values["raw"][0, index]), 4),
             "raw_threshold": round(float(thresholds[index]), 4)}
            for index, label in enumerate(LABELS)]


def classify(audio, device="cpu", policy="f1", encoder=None,
             directory=DEFAULT, compare_baseline=False):
    bundle = load_candidate(directory)
    baseline_bundle = correction = None
    if compare_baseline:
        baseline_bundle = predict_app.load_bundle()
        correction = predict_app.load_correction(
            baseline_bundle, predict_app.METADATA, predict_app.WEIGHTS,
            predict_app.PLAN, predict_app.CORRECTION)
    encoder = encoder or HfMaestEncoder(device=device)
    feature = encoder.extract(audio)
    details = predict_vector(bundle, feature, policy)
    warning = ("Fixed experimental candidate; not yet confirmed on new songs. "
               "Scores are estimates, not guaranteed confidence percentages.")
    if policy == "precision_target":
        warning += " Selective policy suppresses pop and ambient and has low coverage."
    result = {
        "audio": str(Path(audio)),
        "analyzed": "Exactly the first 30 seconds of an input at least 30 seconds long",
        "model": "Discogs-MAEST block 7 plus four logistic-regression heads",
        "candidate_id": bundle["candidate_id"], "status": bundle["status"],
        "policy": policy, "predicted_tags": [d["label"] for d in details if d["selected"]],
        "scores": details, "warning": warning,
    }
    if compare_baseline:
        baseline_details = predict_app.predict_vector(
            baseline_bundle, feature, policy, "weight_corrected", correction)
        result["baseline"] = {
            "application_version": predict_app.APP_VERSION,
            "score_mode": "weight_corrected", "policy": policy,
            "predicted_tags": [d["label"] for d in baseline_details if d["selected"]],
            "scores": baseline_details, "shared_feature": True,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--policy", choices=POLICIES, default="f1")
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT)
    parser.add_argument("--compare-baseline", action="store_true")
    args = parser.parse_args()
    print(json.dumps(classify(args.audio, args.device, args.policy,
                              directory=args.candidate_dir,
                              compare_baseline=args.compare_baseline), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, IndexError) as error:
        raise SystemExit(f"Could not classify audio: {error}") from None
