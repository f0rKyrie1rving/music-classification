# Windows desktop demo — v1.2 confirmed candidate

Version 1.2 uses the fixed candidate confirmed against the v1.1 application on
372 new documented-project tracks from 250 artists. Pop/ambient heads and
thresholds are the exact evaluated files; electronic/rock behavior is preserved.
**Compare v1.1** switches the table and selected tags to the old model's result
using the same audio features, without analyzing the audio again. The original
models and frozen research records are retained.

In the fixed confirmation, pop/ambient false-positive cases fell from 152 to
124 (18.42%), with missed focus labels increasing from 59 to 61. Micro F1 rose
from 0.619910 to 0.637002, and average Brier probability error decreased by 2.72%.
All prespecified point, coverage and confidence gates passed, with independent
verification. The results concern the same source dataset and its proxy labels;
they are not human-certified accuracy or a guarantee for arbitrary music.

See the [confirmation report](https://github.com/f0rKyrie1rving/music-classification/blob/v1.2.0/docs/application_candidate_validation/SUMMARY_ZH.md)
and [application integration record](https://github.com/f0rKyrie1rving/music-classification/blob/v1.2.0/docs/APPLICATION_V1_2.md). Ambient precision
remains low (36.7% relative to source tags); this remains an academic prototype.
The older v1.1 correction experiments retain their original scope in the
[v1.1 record](https://github.com/f0rKyrie1rving/music-classification/blob/v1.2.0/docs/APPLICATION_V1_1.md).

## Download and try

1. When the v1.2 release has an installer under **Assets**, download
   `MusicClassification-Setup-1.2.0-win64.exe`.
   **Source code (zip)** and **Source code (tar.gz)** are for developers.
2. Run the installer and open **Music Tagging Demo** from the Start menu.
   The Windows x64 app targets Windows 10 version 1809 or newer, including
   Windows 11; Python, Git, a terminal, and administrator access are not needed.
3. Click **Use included example**, then **Analyze audio**. The example should
   receive the `pop` and `rock` tags; its v1.1 comparison receives `rock`.
   This checks reproducibility, not human genre adjudication. You can also choose a WAV, FLAC, OGG, or MP3 file
   containing at least 30 seconds of audio.

The first analysis downloads about 348 MB of checksum-verified model files
from the official MTG-UPF Hugging Face repository, so internet access is
required. Later analyses reuse those files. Selected audio stays on your
computer; the app analyzes its first 30 seconds and displays four scores,
their thresholds, and the selected tags.

## Unsigned academic release

The installer and executable have no Authenticode signature. Windows may
display an unknown-publisher or SmartScreen warning, and managed computers
may block installation. Do not disable institutional security controls to
run it. Project results and figures are also available in the
[README](https://github.com/f0rKyrie1rving/music-classification#readme).

`SHA256SUMS.txt` is provided for optional download-integrity verification.
It does not authenticate the publisher.

## Uninstall

In **Settings → Apps**, open **Installed apps** (Windows 11) or **Apps &
features** (Windows 10), select **Music Tagging Demo**, and choose **Uninstall**.
This removes the program, shortcuts, downloaded model, and app cache. Audio
selected from other folders is preserved.

Before publication, the Windows workflow must pass the unit suite, packaged
inference, inference from the installed app, and install/uninstall cleanup.
See the [installation guide](https://github.com/f0rKyrie1rving/music-classification/blob/main/docs/WINDOWS_INSTALLER.md)
for details. Source integration alone does not build or publish an installer.
The displayed values remain research estimates, not guaranteed
confidence percentages; this is a research and portfolio prototype.
