"""Validate both portable and installed v1.2 desktop smoke-test payloads.

This is deliberately stdlib-only so release CI can inspect executable output
without importing a second model or trusting its current working directory.
"""

import argparse
import json
import math
from pathlib import Path


VERSION = "1.2.0"
CANDIDATE_ID = "music-candidate-20260927-v1"
WEIGHTS_SHA256 = "70ceca1ab53b9b8d968d6e1ca4664bc78e22b78e2c70def772d495cbbf208036"
LABELS = ("electronic", "pop", "ambient", "rock")
CANDIDATE_THRESHOLDS = (0.4, 0.275, 0.2, 0.275)
BASELINE_THRESHOLDS = (0.4, 0.275, 0.1321, 0.275)
BASELINE_RAW_THRESHOLDS = (0.4, 0.275, 0.375, 0.275)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def score_rows(result, thresholds, raw_thresholds):
    rows = result.get("scores")
    require(isinstance(rows, list), "Missing score table.")
    require([row.get("label") for row in rows] == list(LABELS),
            "Expected exactly the four application labels in order.")
    for row, threshold, raw_threshold in zip(rows, thresholds, raw_thresholds):
        for key in ("score", "raw_score"):
            value = row.get(key)
            require(type(value) in (int, float) and math.isfinite(value)
                    and 0 <= value <= 1, f"Invalid {row['label']} {key}.")
        require(row.get("threshold") == threshold
                and row.get("raw_threshold") == raw_threshold,
                f"Wrong {row['label']} decision threshold.")
        require(type(row.get("selected")) is bool, "Selection must be a boolean.")
    # Decisions use unrounded values in the application; do not re-threshold
    # the four-decimal presentation values here.
    require(result.get("predicted_tags") == [r["label"] for r in rows if r["selected"]],
            "Predicted tags disagree with saved decisions.")
    return dict(zip(LABELS, rows))


def verify(payload, expected_version=VERSION):
    require(expected_version == VERSION, "Verifier does not match this application release.")
    require(isinstance(payload, dict) and payload.get("ok") is True,
            "Desktop smoke inference did not succeed.")
    result = payload.get("result")
    require(isinstance(result, dict), "Desktop smoke output has no result.")
    require(result.get("application_version") == expected_version,
            "Executable reports the wrong application version.")
    require(result.get("candidate_id") == CANDIDATE_ID
            and result.get("model_weights_sha256") == WEIGHTS_SHA256,
            "Executable used a different candidate model.")
    require(result.get("score_mode") == "validated_candidate"
            and result.get("status") == "validated_candidate",
            "Executable did not use the validated candidate.")
    require(result.get("policy") == "f1", "Desktop default policy must be f1.")
    validation = result.get("validation", {})
    require(isinstance(validation, dict) and validation.get("verdict") == "go"
            and validation.get("tracks") == 372 and validation.get("artists") == 250,
            "Candidate validation evidence is missing or changed.")
    candidate = score_rows(result, CANDIDATE_THRESHOLDS, CANDIDATE_THRESHOLDS)
    require(all(row["score"] == row["raw_score"] for row in candidate.values()),
            "Candidate unexpectedly applies a score correction.")
    require(result["predicted_tags"] == ["pop", "rock"],
            "Executable did not reproduce the included-example candidate decisions.")
    baseline = result.get("baseline")
    require(isinstance(baseline, dict), "Desktop must retain the v1.1 comparison.")
    require(baseline.get("application_version") == "1.1.0"
            and baseline.get("score_mode") == "weight_corrected"
            and baseline.get("policy") == "f1"
            and baseline.get("shared_feature") is True,
            "Baseline comparison is not v1.1 on the same extracted feature.")
    old = score_rows(baseline, BASELINE_THRESHOLDS, BASELINE_RAW_THRESHOLDS)
    require(baseline["predicted_tags"] == ["rock"],
            "Executable did not reproduce the included-example v1.1 decisions.")
    require(old["ambient"]["score"] < old["ambient"]["raw_score"],
            "Included-example baseline lacks the v1.1 ambient correction.")
    for label in ("electronic", "rock"):
        require(all(candidate[label][key] == old[label][key]
                    for key in ("score", "raw_score", "threshold", "raw_threshold", "selected")),
                f"Candidate changed the frozen {label} head or decision.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path)
    parser.add_argument("--expected-version", default=VERSION)
    args = parser.parse_args()
    result = verify(json.loads(args.payload.read_text(encoding="utf-8")), args.expected_version)
    print(f"Verified application {result['application_version']}: {result['candidate_id']}; "
          "fixed model, F1 thresholds, and shared-feature v1.1 comparison.")


if __name__ == "__main__":
    main()
