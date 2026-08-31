# Final blind perceptual-check results

Review completed on 2026-08-30. The 40 queries were frozen before final
holdout scoring. For each of four labels, the set deliberately contains five
source-positive and five source-negative tracks. During listening, the reviewer
saw only the queried label and audio, and answered `yes`, `no`, or `uncertain`.

## Results

| Queried label | Yes | No | Uncertain | Source–human agreement* | Model–human agreement* | Model–source agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| electronic | 1 | 8 | 1 | 5/9 (55.6%) | 5/9 (55.6%) | 7/10 (70.0%) |
| pop | 4 | 5 | 1 | 7/9 (77.8%) | 7/9 (77.8%) | 7/10 (70.0%) |
| ambient | 5 | 5 | 0 | 8/10 (80.0%) | 10/10 (100.0%) | 8/10 (80.0%) |
| rock | 6 | 2 | 2 | 5/8 (62.5%) | 7/8 (87.5%) | 8/10 (80.0%) |
| **Overall** | **16** | **20** | **4** | **25/36 (69.4%)** | **29/36 (80.6%)** | **30/40 (75.0%)** |

\* The denominator excludes four `uncertain` answers. Those answers are
retained rather than forced into either class.

Across the 36 decisive queries, the model and reviewer both said `yes` 13
times and both said `no` 16 times. The model said `yes` on four reviewer-`no`
queries, and `no` on three reviewer-`yes` queries. If the single reviewer's
binary decisions are used only as a descriptive reference, this gives 76.5%
precision (13/17), 81.3% recall (13/16), 78.8% F1, and 80.6% agreement.

## Interpretation

The model agreed with the reviewer more often than the source targets did in
this small check. Ambient is the clearest example: model–human agreement was
10/10, while two of those decisions disagreed with the whole-track source
target. Electronic remains difficult: the reviewer heard electronic character
in only one of ten queried excerpts, including just one of five source-positive
examples.

This supports the earlier finding that whole-track source tags do not always
describe the first 30 seconds, and that missing broad tags can make valid model
outputs count as false positives. It does not show that all disagreements are
label errors; the model still disagreed with seven decisive human answers.

## Limits

This is a secondary qualitative audit, not a new headline accuracy estimate:

- the 40 queries are balanced by source target rather than sampled according
  to real-world tag prevalence;
- there is one reviewer, and broad genres have subjective boundaries;
- each query asks about one label rather than requesting complete annotation;
- the sample is only ten queries per label, with four uncertain answers;
- no answer is used to change the evaluated model or its thresholds.

The frozen 239-track source-label metrics in [FINAL_RESULTS.md](FINAL_RESULTS.md)
remain the formal holdout result. This perceptual check explains part of the
annotation mismatch without replacing that result.
