# Expanded acquisition exceptions

This log is written after acquisition failures occur; it does not alter the
frozen track selection, target ontology, model candidates, or test boundary.

The standard downloader requires exactly 30 seconds of decoded source audio.
If a complete MP3 is checksum-verified but falls no more than 0.1 seconds short,
`repair_short_audio.py` pads only the missing tail with zeros.  Its receipt stores
the original and output frame gaps.  Files that are corrupt, incomplete, or more
than 0.1 seconds short are rejected and require a new documented data plan.

The 0.1-second bound is 0.33% of a 30-second excerpt.  This rule exists for
container/codec boundary rounding, not for making genuinely short tracks pass.
Exact repaired IDs and frame counts are stored in
`data/source/expanded_download_status.json` and their acquisition receipts.
