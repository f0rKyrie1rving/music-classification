# Windows desktop demo — v1.1 score correction

Version 1.1 corrects the ambient label's training class-weight offset before
displaying its estimated probability. The corresponding threshold changes with
it, so tag decisions remain unchanged. **Compare original scores** switches the
desktop table back to the original values without analyzing the audio again.
The classifier heads and original research results are retained.

On two previously observed cohorts (239 and 266 tracks), average Brier error
decreased by 6.13% and 6.57%, respectively. All 505 tracks retained the same tags
under both decision policies. These are retrospective probability-error
reductions, not classification accuracy gains or new independent validation.
The 239-track artist-bootstrap interval includes no improvement. See the
[v1.1 evaluation](https://github.com/f0rKyrie1rving/music-classification/blob/main/docs/APPLICATION_V1_1.md)
for the full results and limitations.

## Download and try

1. Under **Assets**, download `MusicClassification-Setup-1.1.0-win64.exe`.
   **Source code (zip)** and **Source code (tar.gz)** are for developers.
2. Run the installer and open **Music Tagging Demo** from the Start menu.
   The Windows x64 app targets Windows 10 version 1809 or newer, including
   Windows 11; Python, Git, a terminal, and administrator access are not needed.
3. Click **Use included example**, then **Analyze audio**. The example should
   receive the `rock` tag. You can also choose a WAV, FLAC, OGG, or MP3 file
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
for details. The corrected values remain experimental estimates, not guaranteed
confidence percentages; this is a research and portfolio prototype.
