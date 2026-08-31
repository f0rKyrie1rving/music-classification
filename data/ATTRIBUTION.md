# Audio attribution

These learning samples were obtained from the [MTG-Jamendo Dataset](https://github.com/MTG/mtg-jamendo-dataset), using its official `audio-low` mirror at `cdn.freesound.org/mtg-jamendo/`.

The dataset's original `audio_licenses.txt` records each of the following works under [Creative Commons Attribution 3.0 Unported (CC BY 3.0)](https://creativecommons.org/licenses/by/3.0/). Retain the credits, links, license notice and description of modifications when sharing the excerpts. No endorsement by the artists is implied.

| Work | Artist | Original track | Local original |
| --- | --- | --- | --- |
| Бойня | Radio Noiseville | [Jamendo track 543400](https://www.jamendo.com/track/543400) | `raw/543400.low.mp3` |
| Little Place | Sean T Wright | [Jamendo track 207501](https://www.jamendo.com/track/207501) | `raw/207501.low.mp3` |
| Back In The Days | Chill Carrier | [Jamendo track 913702](https://www.jamendo.com/track/913702) | `raw/913702.low.mp3` |

## Modifications

The downloaded MP3 files are the dataset's lower-bitrate mono versions, and their bytes are unchanged. Each WAV preview is a separate excerpt from 30 to 60 seconds of the decoded track, resampled to 22,050 Hz mono, 16-bit PCM. No gain normalization was applied. The preview filenames and cryptographic hashes are recorded in `sample_manifest.json`.

## Provenance

The exact original attribution text, official per-file SHA-256, track metadata, genre tags, archive URL, byte range, retrieval time and source Git blob identifiers are retained in `sample_manifest.json`.

These are demonstrations for audio loading, not a representative benchmark or an evaluation split. Tags are the dataset's track-level annotations; they are not predictions from this project and do not necessarily describe every short excerpt.

## Public-release scope

Only `previews/track_0207501_30s.wav` ("Little Place" by Sean T Wright) is
included in the public release. It is provided under CC BY 3.0 as the default
quick-start input. The other two source files and previews remain local.
