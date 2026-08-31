# Fixed feature comparison: 2026-08-28

This protocol was written before extracting the additional features or fitting
the candidate. The existing baseline validation and test results were already
known. This is an exploratory development comparison, not a new blind test or
a formal external preregistration.

## Question and scope

Does one small, fixed set of additional acoustic descriptors help a linear
multi-label model rank the four genre tags on the existing validation subset?

Exactly two feature sets are compared: the existing 26 MFCC statistics and
those same 26 plus the 20 descriptors below. No search over feature bundles,
MFCC counts, classifiers, regularization strengths, or random seeds is planned.
No interface, new audio acquisition, cloud upload, or replacement of the
existing demo model is part of this experiment.

## Fixed design

- Use the frozen manifest: 300 training and 90 validation tracks only.
- Extract additional features, fit models, and predict only these 390 tracks.
  Test audio is not loaded or predicted. The existing MFCC cache contains
  all splits, but only rows belonging to train/validation are selected.
- Preserve artist separation, original tags, background tracks, source audio,
  preprocessing, and all existing baseline artifacts.
- Use the identical classifier: training-only StandardScaler, then
  OneVsRestClassifier(LogisticRegression(C=1, class_weight="balanced",
  max_iter=2000, random_state=2026)).
- Primary descriptive comparison: validation Macro Average Precision (AP),
  which does not require choosing a label threshold.
- Secondary: validation Micro-F1, Macro-F1, per-label AP/F1, precision, recall,
  and counts of false positives/negatives.
- Select each model's label thresholds using the same existing 0.10–0.90 grid,
  step 0.05, maximizing label F1 and preferring ties nearest 0.5. These F1
  values are evaluated on the same validation set used for threshold selection
  and are optimistic development measurements, not independent test results.
- Also report the fixed 0.5 threshold as a diagnostic, not a newly tuned model.
- Record results even if worse. Do not automatically replace the existing
  baseline or claim significance/generalization from a small validation set.

## Additional features (20)

All audio is processed with the existing first-30-second, mono, 22,050 Hz,
2,048-point Hann window, 512-sample hop, non-centered framing rules.

| Descriptor | Summary | Count | Intended information and limitation |
| --- | --- | ---: | --- |
| Magnitude-weighted spectral centroid | Mean, population SD | 2 | Frequency balance; not musical pitch |
| 85% cumulative **power** rolloff frequency | Mean, population SD | 2 | Distribution of spectral power; not a genre label |
| Spectral flatness of power | Mean, population SD | 2 | Noise-like versus concentrated spectrum; not recording quality |
| Positive log-Mel spectral flux/onset strength | Mean, population SD | 2 | Changes between adjacent frames; not BPM or a full rhythm model |
| 12-bin chroma | Temporal mean per bin | 12 | Pitch-class distribution; not chord recognition or harmonic progression |

Rolloff uses the power spectrogram explicitly. Flatness uses that same power
with exponent 1 and a numerical floor of 1e-10. Onset strength uses lag=1,
max_size=1, no detrending or centering, and a frequency mean. Chroma uses a
power STFT, fixed A440 tuning offset 0, 12 bins C through B, and per-frame
infinity normalization. It is not transposition invariant and may capture
irrelevant key differences; more features can also overfit this small sample.
The first 26 candidate values must equal the baseline extractor's values.

Feature order and source hashes are recorded in `feature_comparison_plan.json`
before the experiment. Installed dependency versions remain unchanged.

## References

- [librosa spectral feature implementation](https://librosa.org/doc/main/_modules/librosa/feature/spectral.html)
- [librosa onset strength](https://librosa.org/doc/0.11.0/generated/librosa.onset.onset_strength.html)
- [librosa chroma tutorial](https://librosa.org/doc/latest/auto_tutorials/03-advanced/plot_chroma.html)

The installed librosa 1.0.0 signatures and implementation were also inspected;
the code and environment lock file define the exact run.
