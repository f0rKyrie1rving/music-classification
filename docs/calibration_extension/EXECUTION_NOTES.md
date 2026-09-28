# Implementation history

The initial `outputs/calibration_extension/20260927_v1` completed the 120 refits.
Before inspecting its numerical outcomes or scoring the retrospective cohort, an
independent code review identified missing provenance-chain guards for the
retrospective cache and summary. The implementation was strengthened to freeze the
existing 266-track manifest/MAEST cache before refits, verify the separate MERT
receipt → extraction freeze → audit → manifest/audio chain before scoring, and verify
retrospective result hashes before summarizing.

No methods, seeds, splits, optimization settings, outcomes or selection rules were
changed. The first run is retained as an implementation-development artifact, with
its original source snapshot. The strengthened implementation is frozen and rerun
under `outputs/calibration_extension/20260927_v2`; only v2 is the reported analysis.
The two executions are not two independent experiments. Since live code changed,
v1's source guard deliberately no longer matches live code; use its snapshot if
inspecting its original implementation. All five earlier research stages are intact.
