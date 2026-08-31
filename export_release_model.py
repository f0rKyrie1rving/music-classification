"""Export the frozen sklearn heads as a safe, deterministic NumPy artifact."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs/final_maest/final_broad_genre_model.joblib"
WEIGHTS = ROOT / "artifacts/final_heads.npy"
METADATA = ROOT / "artifacts/final_heads.json"
LABELS = ("electronic", "pop", "ambient", "rock")
FEATURE_WIDTH = 2_304
REPRESENTATION = "maest_block07_cls_dist_signal_mean"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(source=SOURCE, weights=WEIGHTS, metadata=METADATA):
    """Convert the trusted local joblib bundle without changing its arithmetic."""
    bundle = joblib.load(source)
    if tuple(bundle.get("labels", ())) != LABELS or len(bundle.get("models", ())) != 4:
        raise ValueError("Unexpected label order or number of final heads.")
    if len(bundle.get("selected", ())) != 4:
        raise ValueError("Missing final head selections.")

    means, scales, coefficients, intercepts = [], [], [], []
    for label, selected, pipeline in zip(
            LABELS, bundle["selected"], bundle["models"], strict=True):
        if selected.get("label") != label or selected.get("representation") != REPRESENTATION:
            raise ValueError("Unexpected final representation selection.")
        scaler = pipeline.named_steps.get("standardscaler")
        head = pipeline.named_steps.get("logisticregression")
        if scaler is None or head is None or not np.array_equal(head.classes_, [0, 1]):
            raise ValueError("Unexpected final sklearn pipeline.")
        means.append(scaler.mean_); scales.append(scaler.scale_)
        coefficients.append(head.coef_[0]); intercepts.append(float(head.intercept_[0]))

    parameters = np.stack((means, scales, coefficients)).astype("<f8")
    if parameters.shape != (3, 4, FEATURE_WIDTH) or not np.isfinite(parameters).all():
        raise ValueError("Invalid exported parameter matrix.")
    if np.any(parameters[1] <= 0):
        raise ValueError("Scaler contains a non-positive scale.")

    weights.parent.mkdir(parents=True, exist_ok=True)
    with weights.open("wb") as file:
        np.save(file, parameters, allow_pickle=False)
    record = {
        "format_version": 1,
        "artifact": "four standardized binary logistic-regression heads",
        "labels": list(LABELS),
        "representation": REPRESENTATION,
        "feature_width": FEATURE_WIDTH,
        "parameter_order": ["mean", "scale", "coefficient"],
        "intercepts": intercepts,
        "thresholds": bundle["thresholds"],
        "selected": bundle["selected"],
        "precision_supported": bundle["precision_supported"],
        "evaluation_scope": bundle["evaluation_scope"],
        "evaluation_plan_sha256": bundle["plan_sha256"],
        "encoder": {
            "model_id": "mtg-upf/discogs-maest-30s-pw-129e-519l",
            "revision": "6c35f32a350f74351870937d5ae0bae1d898d1df",
            "hidden_state_index": 7,
        },
        "weights_file": weights.name,
        "weights_sha256": sha256(weights),
        "source_joblib_sha256": sha256(source),
        "license": "CC-BY-NC-4.0; see ../MODEL_LICENSE.md",
    }
    metadata.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {weights.relative_to(ROOT)} ({weights.stat().st_size:,} bytes)")
    print(f"SHA-256 {record['weights_sha256']}")
    return record


if __name__ == "__main__":
    export()
