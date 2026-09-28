# One fixed candidate-versus-application confirmation

2026-09-28. This protocol is fixed before any new archive headers, audio,
features or model predictions from the selected confirmation cohort. The user
authorized the next fully automatic validation. No fitting, calibration,
threshold changes, candidate selection or manual listening labels occur here.

## Frozen comparison

Primary: the standalone music-candidate-20260927-v1, default f1 decisions, versus
the actual application v1.1 with weight-corrected displayed probabilities and
its original unrounded raw-score tag decisions. Bind every candidate bundle file,
the real application weights/metadata/correction, both predictors and the pinned
MAEST extractor/model/runtime by hash. The candidate metadata SHA-256 is
8c335d7140a6b5ea72b6be43c087c72ce917ba352a9f112fd2774464d6a79b12;
its weights SHA-256 is 70ceca1ab53b9b8d968d6e1ca4664bc78e22b78e2c70def772d495cbbf208036.

Use the exact same float32 cached MAEST vector for both models, promoted to
float64 for the actual application's row-wise arithmetic. Verify full precision
and displayed interfaces. Electronic/rock scores and decisions must be exactly
equal. The secondary precision_target policy must have all decisions equal;
its focus displayed probabilities may differ. It cannot replace the primary.

## Frame, selection and exposure

Use the existing verified official MTG-Jamendo metadata at commit
cafd8e20c265ed84f1e61f1c875327971f43a62f, with exact Git blobs in config. Fixed
archives are 12 through 19 inclusive. Require duration at least 30 seconds,
official full MP3 checksum and CC BY/BY-SA/BY-NC/BY-NC-SA license metadata.
Exclude every historically exposed track AND artist ID, including failed and
unselected earlier candidates: 2,195 tracks / 1,008 artist IDs. Check local audio
inventory for unaccounted exposure before freezing; unresolved exposure stops.

The previously assessed eligible frame is 640 tracks / 288 artists. Stop on any
unexpected frame identity/count discrepancy, never silently expand or reduce it.
Hash-order artists by the fixed sample_seed and select the first 250; independently
hash-order their songs and take up to two per artist. Keep exact order. There are
no label quotas or score-based filters. Low positive support will yield incomplete
evidence, not a new seed, extra artists, a replacement or a wider source range.

Freeze selected IDs and source records before indexing. Archive offsets are later
resolved through complete checked TAR header scans and separately hash-bound to
the frozen selection; the original selected_manifest is never edited. The entire
640-track / 288-artist frame is added to future exclusions, even if acquisition or
validation fails. New boundary: 2,835 tracks / 1,296 artists.

Fresh means absent from documented project use, not a wholly independent dataset
or guaranteed unseen music to the pretrained encoder. Artist aliases, external
use and MAEST pretraining overlap remain unknown. Source tags are unchanged proxy
labels, not human-certified exhaustive musical truth.

## Acquisition and missing data

Use exact bounded HTTP 206 byte ranges. Three attempts per range, 8-second connect
timeout, 25-second request timeout, with fixed 1/2/3-second retry waits; keep attempt
receipts. Read only 512-byte TAR headers while indexing, persist every verified
header position, reject duplicate members, changed archive size or bad geometry.
At most three indexing invocations per archive are allowed for recorded infrastructure
recovery. No model scores are computed during acquisition or extraction.

Use four indexing workers and four download workers. Reuse the audited downloader
and its 1,310,720-byte maximum per track. Decode exactly the first 30 seconds to
mono 22,050-Hz PCM16 WAV, preserving licenses, range and checksum receipts.
Partial prefixes are never called full-file SHA-256 verified. Reject insufficient,
silent or nonfinite audio. No arbitrary truncation/padding, mirror switch, track
replacement or unlogged download retry is allowed.

Each selected track gets a durable started/success/failure checkpoint. A terminal
failure is never retried. An interrupted started download becomes a documented
failure on resume. Per-range retries are solely infrastructure handling and use
the same fixed rule for all tracks. Successful cached work is verified and reused.
Acquisition must finish recording ALL selected tracks before extracting/scoring.

MAEST extraction uses CPU and the existing frozen runtime. First reproduce the
original local development control vector (absolute tolerance 1e-5). Persist every
successful/failed feature checkpoint; no outcome-based retries. Model predictions
are computed only after the complete observed cohort and all failures are fixed.
Both models see exactly the same successful rows in selected-manifest order.

## Outcomes and fixed gates

Primary label focus is pop+ambient. Count TP/FP/FN as song-label cases. Report all
four labels, both fixed policies, coverage, Brier, binary log loss, AP, precision,
recall and F1. Compare candidate minus baseline on exactly aligned rows. A positive
classification label is an observed source-tag target; do not relabel disagreements.

Operational POINT gate: combined focus FP reduction at least 10%; each focus FP
nonincrease; micro recall loss at most .03; each label recall loss at most .05;
micro F1 nondecrease; macro Brier and log loss nonincrease (1e-12 metric tolerance).
If baseline focus FP is zero, relative reduction is undefined and cannot pass.
Exact unchanged-label controls and complete provenance are integrity requirements;
violations stop the run rather than become an unfavorable model result.

Evidence coverage requires BOTH observed/selected track and artist fractions >=.8,
at least 250 observed tracks and 200 observed artists; for EACH focus label at least
30 positive and 30 negative tracks AND 20 positive and 20 negative artists.
Positive/negative artist counts may overlap because an artist can have both kinds
of songs. Failures are reported by reason and source, never imputed as model errors.

Use one paired artist-cluster percentile bootstrap: 2,000 draws, fixed seed
2026092805, sample observed artist IDs with replacement, retain all their observed
tracks and all labels, and compare both models in the SAME draw. Report pointwise
95% intervals for delta macro Brier/log loss, micro F1/recall, focus FP cases per
track and focus relative FP change. The FP rate is FP cases/Ntracks (can exceed 1),
not FP/(FP+TN). Count-equivalent differences multiply the draw rate difference by
the original observed track count, avoiding variable draw-size count confusion.
Draws with zero baseline FP have undefined relative change; retain their count and
do not resample them until favorable. Other finite metrics keep all draws.

Formal GO requires adequate coverage, every point condition and BOTH predefined
confidence conditions: the upper 95% interval bound for focus FP cases-per-track
delta is below zero AND the upper bound for macro Brier delta is below zero.
This intersection supports these two prespecified benefits together; it does not
claim adjusted significance for every secondary metric, or uncertainty-adjusted
noninferiority of each recall/F1 guard. No interval or metric is chosen afterward.

Verdict priority: inadequate coverage/support => incomplete_evidence; else any
point failure => not_passed; else both confidence conditions => go; otherwise
promising. Always retain all point, uncertainty and coverage results, including
failed conditions. A go is evidence to consider a release, never an automatic
installer replacement or broad guarantee about arbitrary music.

Intervals are conditional on these fixed models and observed source cohort.
They do not account for historical adaptive research, missing-track bias, label
omissions, artist aliases, pretraining exposure, the large finite-frame sampling
fraction (no finite-population correction) or within-artist song selection.

## Records and stopping

Preserve original source/data/model artifacts and all prior results. Keep a new
freeze, source snapshots, selection, complete header/index receipts, acquisition
failures, cache hashes, row-aligned predictions, bootstrap draws, public summaries
and an independent arithmetic/provenance verification. Never rerun a completed
evaluation to select a favorable seed, cohort or threshold. If the result fails
or is uncertain, report that result and stop this candidate's confirmation.

No training code runs here. No further new-song frame, model revision, application
replacement, release, external sharing or listening-answer submission is part of
this single confirmation. All statuses and limits are reported plainly.
