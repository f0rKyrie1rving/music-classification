"""Predict multiple genre tags with the locally trained MFCC baseline."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib

from features import FEATURE_NAMES, extract_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    path = root / "models/mfcc_baseline.joblib"
    if not path.exists():
        parser.error("No trained model is available. Complete data preparation and run train.py first.")
    # Load only a model you created yourself or obtained from a trusted source.
    bundle = joblib.load(path)
    feature_hash = hashlib.sha256((root / "features.py").read_bytes()).hexdigest()
    if bundle["feature_names"] != FEATURE_NAMES or bundle["feature_source_sha256"] != feature_hash:
        parser.error("Feature code differs from the code used to train this model.")
    values = extract_features(args.audio)
    scores = bundle["model"].predict_proba(values[None, :])[0]
    results = [dict(label=label, score=round(float(score), 4), threshold=round(float(threshold), 4),
                    selected=bool(score >= threshold))
               for label, score, threshold in zip(bundle["labels"], scores, bundle["thresholds"], strict=True)]
    print(json.dumps(dict(audio=str(args.audio), analyzed="First up to 30 seconds of the input file",
                         predicted_tags=[r["label"] for r in results if r["selected"]], scores=results,
                         warning="Experimental model; scores are not calibrated confidence."), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Could not classify audio: {error}") from None
