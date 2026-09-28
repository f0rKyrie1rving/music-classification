"""Current application v1.2: the exact candidate confirmed on 372 new project songs."""

import argparse
import hashlib
import json
from pathlib import Path

from application_release import APP_VERSION, ROOT, load_release
import candidate_predict
import predict_app
from maest_hf_features import HfMaestEncoder
from predict_maest import POLICIES


def load_application(root=ROOT):
    """Load only the accepted candidate, including its immutable construction receipts."""
    root = Path(root).resolve()
    release = load_release(root)
    bundle = candidate_predict.load_candidate(root / release["candidate_directory"])
    if bundle["candidate_id"] != release["candidate_id"]:
        raise ValueError("Application candidate identity does not match the release receipt.")
    return release, bundle


def _load_baseline(root, candidate):
    """Bind the comparison view to the actual, frozen v1.1 model and correction."""
    names = ("artifacts/final_heads.json", "artifacts/final_heads.npy",
             "experiments/final_holdout_plan.json", "artifacts/score_correction.json",
             "app_version.txt")
    # The original candidate receipt binds the model, correction and version.
    # The frozen model metadata separately binds its evaluation plan.
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"The v1.1 comparison artifact is missing: {name}.")
        expected = candidate["baseline_reference"].get(name)
        if expected is not None and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"The v1.1 comparison artifact checksum changed: {name}.")
    metadata, weights, plan, correction_path, version_path = [root / n for n in names]
    baseline = predict_app.load_bundle(metadata, weights)
    correction = predict_app.load_correction(baseline, metadata, weights, plan, correction_path)
    return baseline, correction, version_path.read_text(encoding="utf-8").strip()


def classify(audio, device="cpu", policy="f1", encoder=None,
             compare_baseline=False, root=ROOT):
    """Analyze audio once, applying the accepted model and optional v1.1 comparison."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown prediction policy: {policy}.")
    root = Path(root).resolve()
    release, bundle = load_application(root)
    if compare_baseline:
        baseline, correction, baseline_version = _load_baseline(root, bundle)
    if encoder is None:
        encoder = HfMaestEncoder(device=device)
    feature = encoder.extract(audio)
    details = candidate_predict.predict_vector(bundle, feature, policy)
    warning = "Research estimates, not guaranteed confidence percentages. Pop and ambient can still produce false positives."
    if policy == "precision_target":
        warning += " Selective policy suppresses pop and ambient and has low coverage."
    result = {
        "audio": str(Path(audio)),
        "analyzed": "Exactly the first 30 seconds of an input at least 30 seconds long",
        "model": "Discogs-MAEST block 7 plus four logistic-regression heads",
        "application_version": APP_VERSION,
        "candidate_id": release["candidate_id"],
        "model_weights_sha256": release["model_weights_sha256"],
        "status": "validated_candidate",
        "score_mode": "validated_candidate",
        "validation": release["validation"],
        "policy": policy,
        "predicted_tags": [d["label"] for d in details if d["selected"]],
        "scores": details,
        "warning": warning,
    }
    if compare_baseline:
        old = predict_app.predict_vector(baseline, feature, policy, "weight_corrected", correction)
        result["baseline"] = {
            "application_version": baseline_version,
            "score_mode": "weight_corrected",
            "policy": policy,
            "predicted_tags": [d["label"] for d in old if d["selected"]],
            "scores": old,
            "shared_feature": True,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--policy", choices=POLICIES, default="f1")
    parser.add_argument("--compare-baseline", action="store_true", help="Also show v1.1, using the same audio features.")
    args = parser.parse_args()
    print(json.dumps(classify(args.audio, args.device, args.policy,
                              compare_baseline=args.compare_baseline), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, IndexError) as error:
        raise SystemExit(f"Could not classify audio: {error}") from None
