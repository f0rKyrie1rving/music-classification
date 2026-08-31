"""Tag the first 30 seconds of an audio file with the final local MAEST model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from maest_hf_features import FEATURE_WIDTH, HfMaestEncoder


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "artifacts/final_heads.npy"
METADATA = ROOT / "artifacts/final_heads.json"
PLAN = ROOT / "experiments/final_holdout_plan.json"
LABELS = ("electronic", "pop", "ambient", "rock")
REPRESENTATION = "maest_block07_cls_dist_signal_mean"
POLICIES = ("f1", "precision_target")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bundle(metadata_path=METADATA, weights_path=WEIGHTS):
    """Load and validate the non-executable release artifact."""
    metadata_path, weights_path = Path(metadata_path), Path(weights_path)
    if not metadata_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("Packaged final MAEST classifier heads were not found.")
    bundle = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {"format_version", "artifact", "labels", "representation", "feature_width",
                "parameter_order", "intercepts", "thresholds", "selected",
                "precision_supported", "evaluation_scope", "evaluation_plan_sha256",
                "encoder", "weights_file", "weights_sha256", "source_joblib_sha256",
                "license"}
    if set(bundle) != required or bundle["format_version"] != 1:
        raise ValueError("Final MAEST artifact metadata has unexpected fields.")
    if (tuple(bundle["labels"]) != LABELS or bundle["feature_width"] != FEATURE_WIDTH
            or bundle["representation"] != REPRESENTATION
            or bundle["parameter_order"] != ["mean", "scale", "coefficient"]):
        raise ValueError("Final MAEST label or representation metadata changed.")
    if bundle["weights_file"] != weights_path.name or bundle["weights_sha256"] != sha256(weights_path):
        raise ValueError("Final MAEST artifact checksum mismatch.")
    if len(bundle["selected"]) != len(LABELS):
        raise ValueError("Final MAEST selection count changed.")
    for label, selected in zip(LABELS, bundle["selected"], strict=True):
        if selected.get("label") != label or selected.get("representation") != REPRESENTATION:
            raise ValueError("Final MAEST representation differs from the evaluated system.")
    if not PLAN.is_file() or bundle["evaluation_plan_sha256"] != sha256(PLAN):
        raise ValueError("Final MAEST model does not match the frozen evaluation plan.")
    for policy in POLICIES:
        values = np.asarray(bundle["thresholds"].get(policy), dtype=float)
        if values.shape != (len(LABELS),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid thresholds for policy: {policy}.")
    if not np.allclose(bundle["thresholds"]["f1"], [0.4, 0.275, 0.375, 0.275]):
        raise ValueError("Primary thresholds differ from the final evaluated system.")
    parameters = np.load(weights_path, allow_pickle=False)
    intercepts = np.asarray(bundle["intercepts"], dtype=float)
    if (parameters.shape != (3, len(LABELS), FEATURE_WIDTH)
            or intercepts.shape != (len(LABELS),)
            or not np.isfinite(parameters).all() or not np.isfinite(intercepts).all()
            or np.any(parameters[1] <= 0)):
        raise ValueError("Final MAEST classifier parameters are invalid.")
    bundle["means"], bundle["scales"], bundle["coefficients"] = parameters
    bundle["intercepts"] = intercepts
    return bundle


def predict_vector(bundle, feature, policy="f1"):
    """Apply the four saved heads to one already extracted MAEST vector."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown prediction policy: {policy}.")
    feature = np.asarray(feature, dtype=float)
    if feature.shape != (FEATURE_WIDTH,) or not np.isfinite(feature).all():
        raise ValueError(f"Expected one finite MAEST feature vector of width {FEATURE_WIDTH}.")
    logits = np.sum(
        ((feature - bundle["means"]) / bundle["scales"]) * bundle["coefficients"], axis=1
    ) + bundle["intercepts"]
    scores = np.exp(-np.logaddexp(0.0, -logits))
    if scores.shape != (len(LABELS),) or not np.isfinite(scores).all():
        raise ValueError("Final MAEST heads produced invalid scores.")
    thresholds = np.asarray(bundle["thresholds"][policy], dtype=float)
    details = [
        {"label": label, "score": round(float(score), 4),
         "threshold": round(float(threshold), 4), "selected": bool(score >= threshold)}
        for label, score, threshold in zip(LABELS, scores, thresholds, strict=True)
    ]
    return details


def classify(audio, device="cpu", policy="f1", encoder=None,
             metadata_path=METADATA, weights_path=WEIGHTS):
    bundle = load_bundle(metadata_path, weights_path)
    encoder = encoder or HfMaestEncoder(device=device)
    feature = encoder.extract(audio)
    details = predict_vector(bundle, feature, policy)
    warning = "Experimental research model; scores are not calibrated confidence."
    if policy == "precision_target":
        warning += " Selective policy suppresses pop and ambient and has low coverage."
    return {
        "audio": str(Path(audio)),
        "analyzed": "Exactly the first 30 seconds of an input at least 30 seconds long",
        "model": "Discogs-MAEST block 7 plus four logistic-regression heads",
        "policy": policy,
        "predicted_tags": [item["label"] for item in details if item["selected"]],
        "scores": details,
        "warning": warning,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="WAV, FLAC, OGG, or supported MP3 file")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--policy", choices=POLICIES, default="f1",
                        help="f1 is the four-label default; precision_target is selective")
    args = parser.parse_args()
    print(json.dumps(classify(args.audio, args.device, args.policy), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, IndexError) as error:
        raise SystemExit(f"Could not classify audio: {error}") from None
