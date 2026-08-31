# Expanded broad-genre development results

Development run: 2026-08-30. The new 239-track holdout remains sealed.

## What changed

The training set grew from 300 to 903 tracks and development validation from 90
to 303 tracks. Labels now follow a frozen broad-genre ontology, so subgenres
such as `hardrock`, `poprock`, and `techno` contribute to their parent labels.
The old 90 test tracks are excluded because earlier results were observed.

The frozen MERT-v0-public encoder produced all 13 layer representations for
1,206 development tracks. For each label, 5-fold artist-grouped training CV
selected one of 13 layers or the old transformer-layer mean plus one of six
regularized logistic heads. Thresholds came only from training out-of-fold
scores. See `experiments/EXPANDED_MODEL_PROTOCOL.md`.

## Selected heads

| Label | MERT representation | C | Class weight | Mean grouped-CV AP |
| --- | --- | ---: | --- | ---: |
| electronic | layer 6 | 0.001 | balanced | 0.7625 |
| pop | layer 5 | 0.001 | none | 0.4762 |
| ambient | layer 7 | 0.01 | none | 0.4542 |
| rock | layer 6 | 0.001 | none | 0.6163 |

Nearby layers and strong regularization also ranked highly. The choices are not
isolated outliers, and the full candidate table is stored in
`outputs/expanded/development_metrics.json`.

## Development validation

F1-oriented thresholds were 0.525, 0.200, 0.250, and 0.325.

| Label | Positives | Precision | Recall | F1 | AP | Predicted labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| electronic | 119 | 0.640 | 0.597 | 0.617 | 0.681 | 111 |
| pop | 49 | 0.272 | 0.816 | 0.408 | 0.354 | 147 |
| ambient | 70 | 0.359 | 0.471 | 0.407 | 0.387 | 92 |
| rock | 50 | 0.667 | 0.520 | 0.584 | 0.672 | 39 |

Micro precision was 0.437, micro recall 0.590, micro-F1 0.502, macro AP 0.524,
and output coverage 267/303 = 0.881. These results do not meet the provisional
usefulness target.

The precision-target rule was supported by training OOF data only for
`electronic` (threshold 0.75). On validation it emitted 40 predictions with 31
true positives: precision 0.775, recall 0.261. The other three labels were
suppressed. The target therefore also failed.

## Post-outcome diagnosis

This section is exploratory and was produced after validation results were
known. At the best validation prefixes with at least 0.80 precision:

- electronic: 45 predictions, precision 0.800, recall 0.303;
- pop: 5 predictions, precision 0.800, recall 0.082;
- ambient: no prefix of at least five predictions reached 0.80;
- rock: 26 predictions, precision 0.808, recall 0.420.

Learning curves show electronic beginning to saturate, pop and rock still
improving with more data, and ambient remaining nearly flat. Downloading more
audio with the same noisy labels is therefore not a well-supported next step,
especially for ambient. Exact curves and top-k values are stored in
`outputs/expanded/post_expansion_diagnostic.json`.

## Decision

Do not promote this candidate and do not open the new holdout. The next bounded
candidate changes only the pretrained representation to MERT-v1-95M while
keeping the expanded data, ontology, grouped folds, classifier grid, thresholds,
and validation boundary fixed. This isolates whether substantially broader
music pretraining improves pop and ambient. Reaching 0.80 remains a target, not
a promised result.
