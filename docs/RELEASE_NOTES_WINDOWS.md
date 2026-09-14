# Windows desktop demo — first installer release

The fourth source revision adds a desktop interface and an unsigned Windows
installer for the existing four-label academic music-tagging project. Reported
research results, classifier heads, and evaluation protocols are unchanged.

## Download and try

1. Under **Assets**, download `MusicClassification-Setup-1.0.0-win64.exe`.
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
for details. Scores are not calibrated confidence percentages; this is a
research and portfolio prototype.
