# MAEST listening review

Review completed: 2026-08-30. This is a post-outcome qualitative audit, not a
new test score and not a replacement for the source labels.

## Procedure

The review used the 40 cases frozen in
`data/maest_error_listening_review.csv`: five highest-scoring false positives
and five lowest-scoring false negatives for each of electronic, pop, ambient,
and rock under the MAEST F1 policy.

The project author listened to the same first-30-second excerpt used by the
model. For each case, the reviewer saw only the target broad label and heard
the audio; model score, source tags, and error type were not shown during the
decision. Answers were `yes`, `no`, or `uncertain`, with a short audible reason.
All 40 rows were completed.

## Results

For a source-labelled false positive, `yes` means that the supposedly absent
label was nevertheless audible. For a source-labelled false negative, `no`
means that the source label was not audible in the analyzed excerpt.

| Label | False-positive yes / no / uncertain | False-negative yes / no / uncertain |
| --- | ---: | ---: |
| electronic | 4 / 0 / 1 | 0 / 4 / 1 |
| pop | 4 / 0 / 1 | 2 / 1 / 2 |
| ambient | 3 / 2 / 0 | 1 / 2 / 2 |
| rock | 3 / 1 / 1 | 1 / 4 / 0 |
| **Total** | **14 / 3 / 3** | **4 / 11 / 5** |

Across both error types:

- 25/40 cases plausibly reflect source-label incompleteness or a mismatch
  between whole-track tags and the first 30 seconds: 14 audible target labels
  among false positives plus 11 inaudible source labels among false negatives;
- 7/40 cases are more consistent with genuine model errors: three inaudible
  false-positive labels plus four audible false-negative labels;
- 8/40 cases remained uncertain.

Among the 32 clear `yes`/`no` decisions, 25 disagreed with the binary target
derived from source tags. This proportion must not be generalized to the full
dataset because the review deliberately selected extreme error cases rather
than a random sample.

## Interpretation

The audit supports the hypothesis that the apparent error rate combines at
least two problems:

1. uploader tags are not exhaustive multi-label annotations, so a plausible
   broad genre may be audible even when the source does not declare it;
2. track-level tags may describe a later section of a song while the model and
   reviewer analyze only its first 30 seconds.

The pattern is strongest for electronic: four of five false positives sounded
electronic, while four of five false negatives did not sound electronic in the
excerpt. Pop false positives also frequently sounded pop, supporting missing or
overlapping broad labels. Ambient was the most mixed and subjective category.

This does not prove that MAEST is accurate, and the audited answers must not be
used to silently relabel all development rows or recalculate headline metrics.
It does show that downloading more tracks with the same annotation mismatch is
unlikely to solve the evaluation problem by itself.

## Limitations

- One reviewer completed the audit; there is no inter-rater agreement measure.
- The reviewer knew the target label, which is necessary for a one-label query
  but may prime perception.
- Cases were chosen by error severity after development outcomes were known.
- Broad genres overlap, and `yes` does not mean the label is the only or primary
  genre.
- The result evaluates the analyzed excerpt, not the complete track.

## Decision

Keep the source-label metrics unchanged and do not open the holdout yet. Stop
blindly adding same-source data or swapping encoders. The next bounded step is
to freeze one final model policy, then conduct a small, predeclared independent
evaluation that reports both source-tag metrics and a blinded perceptual review.
The portfolio should present label quality and segment-level validity as part
of the research question rather than claiming that the 40-case audit corrected
the model's precision.
