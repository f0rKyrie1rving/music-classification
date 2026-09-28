# Data exposure after this check

- All 266 candidate track IDs and 171 artist IDs in `results/candidate_pool.json` were inspected as metadata in this phase.
- The fixed selection was 232 songs. Exactly 204 songs from 150 artists were scored by both fixed application versions.
- Twenty-eight selected songs were not scored because of persistent DNS acquisition failures. Their failure records were retained; there was no replacement or extra retry.
- All 204 scored songs are now exposed and cannot serve as untouched validation for subsequent model/correction choices. This remains a fixed-model validation result for commit 931c58b.
- To match this project's conservative exclusion convention, a future new-song audit should exclude the whole 266-candidate / 171-artist frame, even though not every song was scored.
- Union with the previous conservative exclusion is 2,195 track IDs and 1,008 artist IDs. Artist aliases, undocumented external exposure and upstream pretraining are still not resolved by these ID counts.

Machine-readable evidence is in `results/exclusions.json`, `results/candidate_pool.json`, `results/selected_manifest.json`, `results/manifest.json`, `results/download_status.json` and `results/public_predictions.json`.
