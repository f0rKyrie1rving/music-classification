# Design rationale and evaluation boundaries

Written on 2026-08-31 to explain the existing implementation and recorded
experiments. This is a retrospective explanation, not a new preregistration,
an ablation result, or a change to any frozen protocol. Future work is tracked
separately in [ROADMAP.md](ROADMAP.md).

## Why a frozen encoder and four logistic heads?

The system reuses a pretrained music representation and fits a small downstream
classifier for each of electronic, pop, ambient, and rock. Keeping the encoder
frozen makes the downstream experiments practical on the recorded local
hardware and limits how many model choices are fitted to the small labeled
subset. It does not demonstrate that fine-tuning would perform worse.

Each head is a training-fitted StandardScaler followed by regularized logistic
regression. In expanded development, each label selects among six declared
settings using artist-grouped training CV. This provides a controlled comparison
of frozen representations without adding a new nonlinear classifier search.
The four heads permit simultaneous labels and an empty label set; they do not
explicitly model dependencies between labels, and their scores need not sum
to one. See the [classifier implementation](../expanded_model.py) and
[expanded protocol](../experiments/EXPANDED_MODEL_PROTOCOL.md).

## Why the seventh block and CLS + DIST + signal mean?

The [MAEST authors' usage example](https://github.com/palonso/MAEST#using-maest-in-your-code)
extracts the seventh Transformer block and stacks its CLS token, DIST token,
and the average of the remaining tokens. The current project follows that
recipe rather than claiming a new pooling method.

In the Hugging Face implementation used here, hidden-state entry 0 is the
embedding output. Therefore `hidden_states[7]` is the output of the seventh
Transformer block, corresponding to zero-based `transformer_block=6` in the
authors' example. Each component has 768 values, so concatenation produces
2,304 features. The two special-token summaries and the signal-token average
retain different aggregations of the representation; this project has not
established separate rhythm, melody, or genre meanings for those components.

The choice was fixed before MAEST development scoring. No local layer or
pooling ablation established that it is optimal for these four labels. The
[frozen MAEST protocol](../experiments/MAEST_HF_PROTOCOL.md) and
[extractor](../maest_hf_features.py) record the index, dimensions, preprocessing,
and integrity checks. MERT's earlier layer search is a different declared
procedure, not evidence that MAEST underwent the same search.

## Why Macro AP, Micro-F1, and per-label operating metrics?

Average precision (AP) summarizes how well a label's scores rank source-positive
examples ahead of source-negative examples without choosing one output
threshold. Macro AP averages the four label AP values equally, making each
label visible even when their frequencies differ. It is not classification
accuracy or a measure of probability calibration.

Micro-F1 pools true positives, false positives, and false negatives across all
track-label decisions, then balances precision and recall. It measures the
thresholded multi-label output, but frequent labels contribute more decisions.
For that reason the reports also retain per-label precision, recall, F1, AP,
prediction counts, and output coverage. See the official definitions of
[AP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
and [F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

The early default MFCC baseline chose thresholds on its validation set. The
later selected-head and expanded experiments instead used training OOF scores
for thresholds. In expanded development, head settings were selected by mean
fold AP per label. Those same selected OOF scores are development data, not an
unbiased final estimate after model and threshold selection. The
[README comparison](../README.md#model-development-path) separates these stages.

Final MAEST fitting used all 1,206 development tracks while retaining the
already selected settings and thresholds. The independent 239-track evaluation
tested that refitted model; it did not provide another tuning opportunity.
The 80% precision target was not achieved. Classifier scores remain explicitly
uncalibrated, including the ambient head fitted with balanced class weighting.
See [FINAL_RESULTS.md](../FINAL_RESULTS.md).

## Why group by artist?

Tracks by one artist can share performance, timbre, production, and recording
characteristics. Randomly splitting their tracks could let the classifier use
those similarities across training and evaluation. Artist grouping reduces
this source of leakage and measures generalization to artists excluded from
the downstream fit.

The expanded training folds check artist separation, and the scaler and heads
are fitted inside each fold. The final fit contains 469 artists and the holdout
89 different artists. This separation does not establish independence of every
recording, eliminate label noise, or prove that the encoder never encountered
related audio during pretraining.

The current final confidence intervals resample tracks, not artist groups.
Within-artist dependence may therefore affect their interpretation. A planned
artist-cluster bootstrap will be reported separately rather than silently
replacing the frozen track-bootstrap analysis. It has not been run as part of
this documentation update.

## Why retain tracks with none of the four target labels?

The task covers four genres rather than every kind of music. Real music outside
those targets supplies relevant negative examples and prevents the task from
implicitly assuming that every input must receive at least one target label.
These examples are neither fabricated silence nor a fifth mutually exclusive
class: their target vector is `[0, 0, 0, 0]` under the stage's label definition.

A missing source tag is still treated as negative for supervised evaluation,
even though it may be an incomplete annotation. The expanded ontology maps
only the declared subgenres to broad labels; it does not use model predictions
or listening answers to fill missing labels. The source tags describe whole
tracks, while the system analyzes only the first 30 seconds. That fixed excerpt
keeps acquisition and preprocessing bounded and consistent but may miss later
musical content. See the [original dataset protocol](../data/DATASET.md) and
[broad-label mapping](../experiments/EXPANDED_MODEL_PROTOCOL.md).

Track output coverage is the fraction of tracks receiving at least one tag.
It is not accuracy, and an empty output is not automatically evidence of model
uncertainty: the four target genres may simply not apply. A future rejection
policy must distinguish that case from an explicit abstention.

## What does pretraining overlap change?

MAEST was pretrained on Discogs style annotations. Its official repository
does not share the track-level pretraining dataset, so this project cannot
audit its overlap with MTG-Jamendo. Potential overlap could make transfer
results optimistic. The supported claim is artist separation within this
project's downstream data, not proven pretraining-disjoint evaluation. See
the [upstream pretraining-data disclosure](https://github.com/palonso/MAEST#pre-training-data)
and the [final protocol](../experiments/FINAL_HOLDOUT_PROTOCOL.md).

The source-tag distribution is also deliberately enriched for the four targets.
Even a future calibrated model would require validation before its probabilities
could be interpreted on arbitrary user uploads. The single-reviewer
[listening audit](../BLIND_REVIEW_RESULTS.md) illustrates disagreements but
does not create a replacement ground truth or correct the formal test metrics.
