# Development and AI-assistance statement

This is an AI-assisted portfolio project directed and reviewed by Chenglin
Song. The statement below records the division of work so that the repository
does not overstate unaided authorship or hide the use of development tools.

## Chenglin Song's work and decisions

- defined the portfolio purpose, time constraints, four-label scope, and
  multi-label user experience;
- decided to prioritize a working research prototype and command-line release
  over a graphical interface;
- studied and explained the distinction between scores and thresholds, and the
  trade-off among precision, recall, F1, and output coverage;
- made all 40 subjective listening-review decisions and wrote the musical
  reasons for those decisions without AI-generated answers;
- tested the prediction command on independently obtained audio and judged
  whether its output was musically plausible;
- reviewed the reported limitations and approved the final portfolio framing.

## How generative AI was used

- drafted and refactored portions of the Python implementation under the
  author's direction, including data checks, feature-extraction wrappers,
  evaluation utilities, and the command-line interface;
- helped debug environment, dependency, download, and model-format issues;
- proposed automated tests and integrity checks that were then executed on the
  local project;
- organized experiment records and helped edit English and Chinese technical
  documentation for clarity.

## Research-integrity safeguards

- No AI-generated audio or labels entered training, validation, or evaluation.
- Reported metrics were computed by the checked-in pipeline from documented
  dataset manifests; they were not invented or estimated by a language model.
- Model choices and thresholds were frozen before the independent holdout was
  acquired and scored. The holdout result was retained even though it missed
  the original precision target.
- Listening answers were not used to retrain or retune the evaluated model.
- Third-party datasets and pretrained encoders are cited and licensed
  separately. AI tools are not listed as authors.

Chenglin Song is responsible for the claims made in this portfolio and should
be able to explain its research question, pipeline, evaluation design, main
results, and limitations.
