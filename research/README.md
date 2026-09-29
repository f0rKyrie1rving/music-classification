# Probability-calibration research

Current application status (2026-09-29): the confirmed fixed candidate is now
integrated into the v1.2 desktop and command-line application. See the
[integration report](../docs/APPLICATION_V1_2.md). Statements below that the
application was unchanged describe the state at those earlier research stages.
Windows packaging, audio inference, installation and uninstall validation have
since passed. The installer is publicly available in the
[v1.2.0 release](https://github.com/f0rKyrie1rving/music-classification/releases/tag/v1.2.0).

This directory preserves the experiments and evidence behind the application.
The calibration study has not yet been extracted into a standalone research
repository; several reproduction procedures still require the documented local
feature caches. Keeping this history here does not imply that a separate project
or a fully self-contained research reproduction package has been published.

The study tests the quality of four independent music-label probabilities relative
to observed MTG-Jamendo tags. The original application and historical artifacts are
preserved. The following stages were completed in order:

| Stage | Question | Results and reproduction |
| --- | --- | --- |
| Local audit and initial calibration | Are the caches usable, and do simple calibrators improve the existing probability estimates? | [Report](../docs/calibration/REPORT.md), [code](calibration/README.md) |
| Calibration-data budget | How does the available calibration data affect results? | [Report](../docs/calibration_budget/REPORT.md), [code](calibration_budget/README.md) |
| Failure diagnosis | What happened in the unfavorable original split? | [Report](../docs/calibration_diagnostics/REPORT.md), [code](calibration_diagnostics/README.md) |
| Conservative calibration | Can internal validation choose smaller corrections? | [Report](../docs/calibration_conservative/REPORT.md), [code](calibration_conservative/README.md) |
| New project-artist validation | Does the fixed primary correction transfer to new documented-project-disjoint songs? | [Report](../docs/calibration_validation/REPORT.md), [code](calibration_validation/README.md) |
| Mechanism, representation transfer and refitting | Which gains come from ambient class weighting, do they recur across MAEST/MERT, and how do 20 refits vary? | [Report](../docs/calibration_extension/REPORT.md), [Chinese summary](../docs/calibration_extension/SUMMARY_ZH.md), [code](calibration_extension/README.md) |
| Fixed application on later new songs | Does the current application's analytical score correction transfer to the next project-disjoint cohort? | [Chinese summary](../docs/application_fresh_validation/SUMMARY_ZH.md), [code](application_fresh_validation/README.md) |
| Automatic development gate | Can nested artist validation select pop/ambient heads and thresholds that reduce false positives without excessive missed labels? | [Chinese summary](../docs/application_auto_improvement/SUMMARY_ZH.md), [code](application_auto_improvement/README.md) |
| Head/rule ablation and training amount | Which changes affect scores versus decisions, and does more exposed training data help on the same outer-held songs? | [Chinese summary](../docs/application_factorial/SUMMARY_ZH.md), [all learning curves](../docs/application_factorial/LEARNING_CURVES.md), [code](application_factorial/README.md) |
| Fixed candidate construction | Can the development procedure produce a runnable fixed candidate while preserving the actual application's electronic/rock behavior? | [Chinese summary](../docs/application_candidate/SUMMARY_ZH.md), [construction code](application_candidate/README.md) |
| Fixed candidate on new project songs | Does the frozen candidate improve on the actual application under predefined error, coverage and uncertainty rules? | [Chinese summary](../docs/application_candidate_validation/SUMMARY_ZH.md), [confirmation code](application_candidate_validation/README.md) |

The fixed-model validation stage scored all 266 prespecified songs from 200 new project artist IDs.
Macro Brier was 0.133726 for raw probabilities, 0.126266 for full sigmoid, and
0.131118 for conservative sigmoid. Both corrections improved the observed point
estimate; the conservative method did not outperform the fixed full-sigmoid comparator.
These results concern the frozen research model and restricted sample, with the
novelty and conditional-interval limitations documented in the report.

The subsequent exploratory extension contains 20 paired artist splits of each of
three frozen representations and two ambient-weighting settings (120 fitted
pipelines). Under ambient-balanced training, full sigmoid improves macro Brier in
all 20 splits for every representation. Without class weighting, its average effect
is slightly adverse for all three. Simple weight-offset and intercept controls
support class-weight-induced probability displacement as an important source of
the earlier gains. This is same-dataset evidence, not cross-dataset confirmation;
the 266-song cohort is now explicitly a retrospective stress check.

For a short explanation, read the [Chinese results](../docs/calibration_validation/SUMMARY_ZH.md)
and [what calibration changes](../docs/CALIBRATION_EXPLAINED_ZH.md).
For any subsequent fresh test, start from the [historical scored-data ledger](../docs/application_fresh_validation/DATA_USE_LEDGER.md)
and the [latest retirement and exclusion rules](application_candidate_validation/protocol.md).
The current [exclusion file](../docs/application_candidate_validation/future_exclusions.json) is
copied from `outputs/application_candidate_validation/20260928_v1/future_exclusions.json`:
2,835 track IDs and 1,296 artist IDs, including the whole latest 640-track frame.
The first automatic development candidate failed its prespecified gate: pop/ambient false
positives fell 4.87%, but micro F1 declined slightly. No final candidate was fitted
or installed, and no further new-song cohort was acquired in that stage.

The subsequent factorial study explicitly retired the observed 239-, 266- and
204-song cohorts into an expanded development pool of 1,915 tracks / 908 artists.
Its fixed combined head/rule procedure reduced focus-label false positives from
776 to 697 (10.18%), with false negatives rising from 261 to 271; micro F1 rose
from 0.605109 to 0.613212. This narrowly passed the development gate. Probability
learning curves improved in the pooled sample for both recipes and all three
prespecified prefix draws, though source-level exceptions remain. Both arms are
refitted recipes; this is not confirmation against the installed application.
No final artifact was fitted and no fresh-song cohort was acquired in that study.
Its then-current exclusion boundary was 2,195 tracks / 1,008 artists.

The subsequent construction stage fitted fixed pop/ambient heads on all 1,915
development rows and copied electronic/rock from the real application. A fixed
four-fold OOF procedure selected pop's original cutoff and ambient's 0.20 cutoff;
its scores are selection diagnostics, not another validation result. The separate
candidate bundle and side-by-side audio entrypoint are available and verified.
The installed application remains unchanged.

The subsequent frozen confirmation acquired and extracted all 372 selected tracks
from 250 new documented-project artist IDs. Focus false positives fell from 152
to 124 (18.42%), with missed focus labels increasing from 59 to 61. Micro F1 rose
from 0.619910 to 0.637002, and macro Brier fell from 0.128989 to 0.125475 (2.72%).
Every fixed point, coverage and confidence gate passed. Paired artist bootstrap
95% intervals for candidate-minus-baseline macro Brier and focus FP cases per
track were [-0.00591445, -0.00119525] and [-0.11570540, -0.03759247]. Exact replay
and independent reconstruction passed. The relative FP-change interval extends
from -26.90% to -9.70%; the 10% gate applies to the point estimate, not a proven
10% lower confidence bound. These are same-source proxy-label results, with no
automatic installation or release. The entire latest eligible frame is retired.
