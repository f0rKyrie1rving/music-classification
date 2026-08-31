# MERT-v1 pilot correction

The first frozen MERT-v1 plan produced no features and no downstream scores.
Its pilot rejected the first verified 30-second WAV because `librosa.load` with
22,050-to-24,000 Hz conversion returned 720,001 samples instead of the expected
720,000. A check of the first ten frozen training files found the same one-sample
surplus for every file. Direct decoding with `soundfile` followed by the pinned
`soxr` resampler returned exactly 720,000 samples for every checked file.

The v2 implementation therefore reads exactly 30 source seconds with
`soundfile`, averages channels, and resamples with `soxr` HQ. It neither trims
nor pads the development inputs; any output other than exactly 720,000 samples
is rejected. The original failed plan remains at
`experiments/mert_v1_plan.json`; v2 uses `experiments/mert_v1_plan_v2.json` and
records the failed plan's hash. No MERT-v1 feature or prediction existed before
this correction.
