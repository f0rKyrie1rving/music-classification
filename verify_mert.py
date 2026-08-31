"""Verify actual cached features, saved predictions and repeatable head fitting."""

import json
import subprocess
import sys

import joblib
import numpy as np

from improve_model import OUT, load_development, operating_metrics
from mert_features import ROOT, MertEncoder, sha256


def main():
    plan, rows, y, _ = load_development()
    ids = np.array([r["track_id"] for r in rows])
    n = len(plan["training_ids"])
    result = json.loads((OUT / "mert_metrics.json").read_text())
    head = joblib.load(OUT / "mert_head.joblib")
    execution = json.loads((ROOT / "experiments/mert_execution_plan.json").read_text())
    cache_path = OUT / "mert_features.npz"
    assert result["cache_sha256"] == sha256(cache_path)
    with np.load(cache_path, allow_pickle=False) as data:
        x = data["x"]
        np.testing.assert_array_equal(data["ids"], ids)
        np.testing.assert_array_equal(data["audio_hashes"], [r["wav_sha256"] for r in rows])
        assert all(str(data[k]) == str(v) for k, v in execution.items())
    assert x.shape == (390, 768) and np.isfinite(x).all()
    assert (x.std(axis=0) > 0).all(), "Constant feature columns need investigation."
    assert len({row.tobytes() for row in x}) == len(rows), "Duplicate complete vectors need investigation."
    np.testing.assert_allclose(head["model"][0].mean_, x[:n].mean(axis=0), rtol=0, atol=1e-12)
    with np.load(OUT / "mert_scores.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["ids"], ids)
        np.testing.assert_array_equal(data["y"], y)
        assert data["oof_scores"].shape == (300, 4)
        scores = head["model"].predict_proba(x[n:])
        np.testing.assert_array_equal(data["validation_scores"], scores)
    for policy, thresholds in head["policies"].items():
        assert operating_metrics(y[n:], scores, thresholds, plan["labels"]) == result["validation"][policy]
        positive = scores >= thresholds
        tp = int((positive & (y[n:] == 1)).sum())
        count = int(positive.sum())
        assert result["validation"][policy]["micro_precision"] == tp / max(count, 1)
        assert result["validation"][policy]["micro_recall"] == tp / int(y[n:].sum())
    encoder = MertEncoder(execution["device"])
    # Fixed positions, not selected for good/bad model scores.
    checked = []
    for i in (0, 149, 389):
        repeated = encoder.extract(ROOT / rows[i]["audio_file"])
        np.testing.assert_allclose(repeated, x[i], rtol=1e-5, atol=1e-6)
        checked.append(dict(track_id=rows[i]["track_id"], max_abs_difference=float(np.max(np.abs(repeated-x[i])))))
    paths = [OUT / f"mert_{name}" for name in ("metrics.json", "scores.npz", "head.joblib")]
    before = {path.name: sha256(path) for path in paths}
    subprocess.run([sys.executable, "improve_model.py", "mert", "--cache", str(cache_path)],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    after = {path.name: sha256(path) for path in paths}
    assert before == after, "Repeated head training changed artifacts."
    load_development()
    report = dict(cache_shape=list(x.shape), unique_track_vectors=len(rows),
        all_feature_columns_nonconstant=True, saved_head_reproduces_scores=True,
        scores_reproduce_metrics=True, final_scaler_train_only=True,
        repeated_feature_checks=checked, repeated_head_fit_byte_identical=True,
        artifact_sha256=after, baseline_unchanged=True, test_prediction_count=0)
    (OUT / "mert_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
