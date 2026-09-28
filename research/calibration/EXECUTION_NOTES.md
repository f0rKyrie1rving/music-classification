# Execution notes

- Original working tree was clean on main at fa09b95ef889be769e3d7add4abdb31288fc1191.
  No filesystem AGENTS.md was found within the project or its ancestor directories.
  Research uses separate research/calibration and outputs/calibration directories in
  the cache-bearing checkout. No historical files or Git branches were changed.
- Run 20260926_v1 froze the protocol and all five splits before new predictions.
  Stage A completed. The primary experiment stopped at its strict identity assertion,
  before saving evaluation metrics. All 12 calibrator optimization calls succeeded.
- Diagnosis: scikit-learn 1.9 retained float32 head coefficients/logits from float32
  features. The calibrator intentionally promoted logits to float64, while the
  assertion compared against float32 scipy expit; maximum difference was
  7.505601140600504e-08 over 964 entries. This was a precision mismatch, not a
  calibration effect. Parameter reconstruction alone did not reproduce it because
  that path already used float64.
- Run 20260926_v2 explicitly promotes cached features to float64 for CPU classifier
  fitting and prediction, as intended by the original protocol. The stored original
  features remain float32 and untouched. A dtype regression check was added. Config,
  protocol and splits are unchanged; the v1 failure and source snapshot are retained.
  No evaluation metric was inspected before this correction.
