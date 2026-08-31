# Formal dataset preparation

This document concerns the classification subset. The three earlier learning
samples in `sample_manifest.json` are separate and are excluded by track ID.

## Task and labels

The four targets are `electronic`, `pop`, `ambient`, and `rock`. These are
original MTG-Jamendo genre tags, not new annotations or mutually exclusive
classes. Tracks keep all their original tags in the manifest. Missing tags are
treated as negatives for this baseline, although uploaders' annotations can be
incomplete or noisy.

Labels were chosen before training from broad styles with substantial coverage
in the official split-0 training data: electronic 10,034 tracks, pop 4,523,
ambient 4,185, and rock 3,995. No model predictions or test scores were used to
choose the labels.

## Sampling protocol

- Preserve the official `split-0` train/validation/test membership; verify zero
  intersections of track IDs and artist IDs across partitions.
- Scan the first 420 TAR headers in each of archive shards `00`, `01`, `02`,
  and `03` for bounded acquisition.
  This is a convenience subset, not a representative random sample of all music.
- Use tracks with a documented Creative Commons license allowing adaptations
  (exclude licenses containing `nd`), with metadata duration at least 30 seconds.
- Select 300 training, 90 validation, and 90 test tracks. Sampling is deterministic
  with seed string `music-portfolio-2026-08-27-v1` and hash ordering of track IDs.
- Limit each artist to three training tracks or two validation/test tracks.
- Reserve 10% of each partition for tracks with none of the four target tags.
  These are still music, not silence or fabricated negative examples.
- Enrich label coverage to at least 50 positives per training label and 15 per
  validation/test label, then fill remaining places from eligible tracks.
  This changes class prevalence; results must not be compared directly with the
  official full-dataset benchmark or claimed as deployment performance.
- Freeze the selected manifest before extracting features or training. Record
  any subsequent exclusions and their technical reasons; never remove a track
  because its prediction is wrong.

## Audio acquisition and integrity

The official low-bitrate MP3 TAR files are accessed using bounded HTTP byte
ranges. TAR headers identify the member and its exact start and size. Usually
only a prefix of a member is needed to decode the first 30 seconds. The
downloader inspects the first 64 KiB, then normally fetches 384 KiB in total,
extending only if necessary. A large leading ID3 tag (for example, embedded
album artwork) can be skipped using a second byte range that starts at the
MPEG audio; its length is parsed according to the
[ID3 specification](https://id3.org/id3v2.4.0-structure). Cached original bytes
plus any separate audio segment are capped at 1.25 MiB per track. Transfer
retries may add network traffic beyond the amount of unique cached bytes.

The downloader requires correct HTTP Content-Range and byte counts. It records
each downloaded segment's byte range and hash, and each derived WAV hash. An official full-MP3
hash is retained for reference, but **a partial MP3 is not claimed to match or
verify the full-file hash**. Full-file verification is marked true only if the
entire member was fetched and its hash actually matches.

Decode the first 30 seconds from the start of each input, average channels to
mono if needed, resample to 22,050 Hz with soxr HQ, and store PCM16 WAV. Do not
normalize the gain. Require exactly 661,500 output samples and finite,
non-negligible audio. Track-level genre labels may not all be audible in this
particular excerpt.

The prefix approach was checked on all three fully downloaded learning MP3s:
the first 30 seconds decoded from 384 KiB prefixes were sample-for-sample equal
to the corresponding full-file decode. This verifies those three comparisons,
not the full-file integrity of every future partial download.

All audio remains local and ignored by Git. Receipts and cached prefixes are in
`data/source/`; selected audio is in `data/training_audio/`. The frozen public
manifest carries the metadata, individual licenses, source references, and
derived-audio hashes needed to identify this particular experiment.

## Completed acquisition audit (2026-08-27)

All 480 selected excerpts passed the checks; none was removed or replaced.
There are 328 distinct artist IDs, with no overlap across partitions. Selected
source segments total 200,384,144 bytes (about 200.4 MB, excluding transfer
retries and separate index/pilot work). Six recordings required skipping large
ID3 tags. Only three selected MP3 members were acquired in full and verified
against official full-file hashes; the other 477 have segment and derived-WAV
hashes, **not** full-file verification. See [audit record](dataset_audit.json).

| Split | Tracks | Artists | electronic | pop | ambient | rock | Multiple target tags | No target tags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 300 | 199 | 136 | 71 | 64 | 69 | 68 | 30 |
| Validation | 90 | 64 | 46 | 16 | 18 | 17 | 16 | 9 |
| Test | 90 | 65 | 35 | 25 | 15 | 22 | 16 | 9 |

Counts refer to the four selected targets; a recording can have other original
genre tags as well. Re-running the deterministic selection reproduced all
480 IDs in order. Source tags, artist/album IDs, splits, licenses, and official
reference hashes were also checked against the retained metadata pool.

## Licensing and attribution

Dataset: [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset), by Dmitry
Bogdanov, Minz Won, Philip Tovstogan, Alastair Porter, and Xavier Serra (2019).
See [the dataset paper](http://hdl.handle.net/10230/42015).

MTG-Jamendo metadata is licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The derived label/metadata manifest retains that license and credits the
dataset. Changes: filtering, selecting four target labels, preserving split-0
membership, adding acquisition records and excerpt paths.

Each recording has its own Creative Commons license, retained verbatim in the
manifest along with creator, title, original track URL, and modification notes.
Do not assume the dataset code license covers either metadata or audio.

The official dataset repository limits the dataset to non-commercial research
and academic use and directs commercial users to obtain authorization from
Jamendo. This portfolio uses that research/academic scope; it does not grant a
commercial license to the data or resulting model. Code licensing and any model
distribution terms must be stated separately before publication. No audio,
model, metadata, or code has been uploaded to GitHub in this preparation step.
