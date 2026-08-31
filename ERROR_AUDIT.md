# Development validation error audit

This is an **exploratory** audit made after the 90-track development validation results were known. It is useful for diagnosing the next data/model step, not for making a fresh performance claim.

## What was counted

The table uses the MERT candidate's F1-oriented thresholds. A false positive means the model emitted a label absent from the source tags; a false negative means a source target tag did not pass its threshold.

| Label | False positives | False negatives | FPs with a declared neighbouring tag |
| --- | ---: | ---: | ---: |
| electronic | 16 | 13 | 3 |
| pop | 18 | 10 | 4 |
| ambient | 17 | 6 | 1 |
| rock | 5 | 8 | 2 |

Ten of 56 false positives contain one of the declared neighbouring tags. Examples include `hardrock` without `rock`, `rocknroll` without `rock`, and `chanson` without `pop`. This is evidence of a taxonomy or annotation issue worth listening to; it does **not** prove that the model is correct. The heuristic mapping is stored in `outputs/improvement/error_audit.json` and is never used to change labels or metrics.

## Fixed listening sample

For each label, take the three highest-scoring false positives and three lowest-scoring false negatives. This produces 24 cases without hand-picking appealing examples.

| Label | Error | Track | Artist — title | Score / threshold | Source tags | Neighbour flag |
| --- | --- | --- | --- | ---: | --- | --- |
| electronic | FP | track_1371700 (`data/training_audio/track_1371700.wav`, local audio, not distributed) | Dj-J@M — The escape | 0.934 / 0.400 | ambient, atmospheric, dance | dance |
| electronic | FP | track_0387401 (`data/training_audio/track_0387401.wav`, local audio, not distributed) | Matti Paalanen — Haunting | 0.843 / 0.400 | ambient, chillout, easylistening | — |
| electronic | FP | track_1391102 (`data/training_audio/track_1391102.wav`, local audio, not distributed) | DeadBunny Laura — Running Away | 0.734 / 0.400 | dance, pop, rock | dance |
| electronic | FN | track_1305603 (`data/training_audio/track_1305603.wav`, local audio, not distributed) | One Dice — Maskenball | 0.088 / 0.400 | darkwave, electronic | — |
| electronic | FN | track_0294403 (`data/training_audio/track_0294403.wav`, local audio, not distributed) | Hozonie — shamann | 0.118 / 0.400 | electronic, techno, trance | — |
| electronic | FN | track_1388000 (`data/training_audio/track_1388000.wav`, local audio, not distributed) | Sean T Wright — Pretty Little | 0.173 / 0.400 | electronic, industrial, rock | — |
| pop | FP | track_1231902 (`data/training_audio/track_1231902.wav`, local audio, not distributed) | Guy Berrier — OUT OF MY HOME 2 | 0.577 / 0.250 | 80s, ambient, classical | — |
| pop | FP | track_0974000 (`data/training_audio/track_0974000.wav`, local audio, not distributed) | Anitek — Seasons | 0.491 / 0.250 | acidjazz, blues, experimental, jazz, rock, triphop | — |
| pop | FP | track_1359901 (`data/training_audio/track_1359901.wav`, local audio, not distributed) | Kaceo — Disque d'or | 0.454 / 0.250 | chanson | chanson |
| pop | FN | track_0037700 (`data/training_audio/track_0037700.wav`, local audio, not distributed) | Mi Rara Coleccion — El silencio me hipnotiza  | 0.007 / 0.250 | electronic, pop | — |
| pop | FN | track_0604700 (`data/training_audio/track_0604700.wav`, local audio, not distributed) | Sean T Wright — B With U | 0.039 / 0.250 | pop, poprock | — |
| pop | FN | track_1115703 (`data/training_audio/track_1115703.wav`, local audio, not distributed) | Podington Bear — Slide | 0.049 / 0.250 | indie, instrumentalpop, pop | — |
| ambient | FP | track_1036200 (`data/training_audio/track_1036200.wav`, local audio, not distributed) | Eon — After the End | 0.875 / 0.300 | alternative, electronic | — |
| ambient | FP | track_0255702 (`data/training_audio/track_0255702.wav`, local audio, not distributed) | The Chill — The Chill - Night on the roof | 0.829 / 0.300 | electronic | — |
| ambient | FP | track_1045800 (`data/training_audio/track_1045800.wav`, local audio, not distributed) | MilanWulf — MilanWulf-Lunacy | 0.816 / 0.300 | drumnbass, dub, dubstep, electronic, house | — |
| ambient | FN | track_0826301 (`data/training_audio/track_0826301.wav`, local audio, not distributed) | Antoniocamel — Travesuras | 0.020 / 0.300 | ambient | — |
| ambient | FN | track_0506002 (`data/training_audio/track_0506002.wav`, local audio, not distributed) | Franck Mouzon — Les oiseaux | 0.070 / 0.300 | ambient, electronic, instrumentalpop, progressive, symphonic | — |
| ambient | FN | track_1231902 (`data/training_audio/track_1231902.wav`, local audio, not distributed) | Guy Berrier — OUT OF MY HOME 2 | 0.087 / 0.300 | 80s, ambient, classical | — |
| rock | FP | track_0160301 (`data/training_audio/track_0160301.wav`, local audio, not distributed) | Cui Bono — Betrayed | 0.907 / 0.300 | hardrock | hardrock |
| rock | FP | track_1115703 (`data/training_audio/track_1115703.wav`, local audio, not distributed) | Podington Bear — Slide | 0.695 / 0.300 | indie, instrumentalpop, pop | — |
| rock | FP | track_0100603 (`data/training_audio/track_0100603.wav`, local audio, not distributed) | The Wavers — Three Minutes to Escape | 0.494 / 0.300 | country, rocknroll, soundtrack | rocknroll |
| rock | FN | track_0004402 (`data/training_audio/track_0004402.wav`, local audio, not distributed) | DIY-note — Raining outside | 0.054 / 0.300 | alternative, electronic, experimental, postrock, rock | — |
| rock | FN | track_0004400 (`data/training_audio/track_0004400.wav`, local audio, not distributed) | DIY-note — Conte écourté | 0.072 / 0.300 | alternative, electronic, experimental, postrock, rock | — |
| rock | FN | track_0040302 (`data/training_audio/track_0040302.wav`, local audio, not distributed) | Melophon — Mr. Ponty | 0.085 / 0.300 | fusion, jazz, rock | — |

## How to listen

Open `data/error_listening_review.csv`. For each case, listen without treating the model prediction as an answer. Enter `yes`, `no`, or `uncertain` under `auditor_hears_label`, then add one short reason. A single listener's judgement remains a portfolio error analysis, not replacement benchmark ground truth.

## What this changes

The automated findings justify two actions already frozen in the expansion protocol: use all eligible training data instead of resampling a small subset, and keep the new test tracks sealed until every model choice is fixed. No label has been changed as a result of this audit.
