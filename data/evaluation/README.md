# Public evaluation package

This package was added after the historical evaluation to make its reported
numbers independently checkable. It contains no audio, pretrained weights,
features or executable model objects.

- `holdout.json`: all 239 track IDs and artist IDs, four broad source targets,
  full-precision saved classifier scores, and decisions for both frozen policies.
  Label order and the original final-plan and score-archive hashes are recorded.
- `original_metrics.json`: the complete original final metrics and 2,000-replicate
  track-bootstrap intervals for both policies, copied as JSON data from the
  retained `outputs/final_maest/holdout_metrics.json`.
- `original_blind_summary.json`: original overall and per-label perceptual-check
  counts. Human answers and query identities are in `../final_blind_review.csv`
  and `../../experiments/final_holdout_plan.json`.
- `development_feature_provenance.json`: the retained development cache metadata,
  excluding only `extraction_seconds`, `resumed_tracks`, and `new_tracks`. It
  retains feature/audio hashes, IDs, model/source hashes, device and versions.

JSON exports preserve values; file formatting need not reproduce the original
local file bytes. Source-artifact hashes identify the historical originals.

From the repository root, install `requirements-evaluation.txt` and run:

```bash
python evaluate_release.py --output outputs/review_audit.json
```

This needs NumPy and scikit-learn, but no encoder, audio acquisition, source pool,
or pre-existing `outputs/`. It verifies every archived point metric and original
interval, plus all original human-agreement counts. It also calculates the
post-hoc checks described in [EVALUATION_REVIEW.md](../../EVALUATION_REVIEW.md).
The reduced subsets do not replace the original 239-track evaluation.

Track/artist IDs and targets are derived from the MTG-Jamendo metadata by
Dmitry Bogdanov, Minz Won, Philip Tovstogan, Alastair Porter and Xavier Serra
(2019), [official dataset](https://github.com/MTG/mtg-jamendo-dataset).
The metadata and derived evaluation table are shared under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Changes include subset selection, the documented broad-label mapping, and the
addition of model scores and decisions. This supplies no license to source audio
or pretrained weights; see [MODEL_LICENSE.md](../../MODEL_LICENSE.md).
