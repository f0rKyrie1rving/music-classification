# Windows desktop release

The Windows release is a per-user, 64-bit desktop application for Windows 10
version 1809 or newer. Reviewers do not need Python, Git, or a terminal.
The release workflow builds with Python 3.12 because all pinned runtime
dependencies publish Windows x64 wheels for that interpreter version.

## Install and run

1. Open the repository's [Releases page](https://github.com/f0rKyrie1rving/music-classification/releases).
2. Download `MusicClassification-Setup-<version>-win64.exe` and the optional
   `SHA256SUMS.txt` verification file from **Assets**. Do not choose **Source
   code (zip)** or **Source code (tar.gz)**: those are source archives.
3. Run the installer. Administrator access is not required.
4. Open **Music Tagging Demo** from the Start menu.
5. Choose a WAV, FLAC, OGG, or MP3 file of at least 30 seconds, or use the
   included attributed example, then select **Analyze audio**.

If the release has no `.exe` asset, a downloadable installer is not available
yet. The [Windows workflow](https://github.com/f0rKyrie1rving/music-classification/actions/workflows/windows-installer.yml)
shows build progress. The interface is in English for academic reviewers.

The first analysis downloads approximately 348 MB of pinned Discogs-MAEST
files from the official MTG-UPF Hugging Face repository. The app verifies their
recorded file sizes and digests before loading them. Later analyses reuse the
verified local copy. The first download requires internet access to Hugging
Face; if it fails, check connectivity and retry. The installer excludes these third-party
weights because the exact upstream model card does not declare their license.

The app reads exactly the first 30 seconds. It does not upload, move, rewrite,
or retain the selected audio. Scores are classifier outputs, not calibrated
confidence percentages.

## Uninstall

Open **Settings → Apps → Installed apps** (Windows 11) or **Apps & features**
(Windows 10), find **Music Tagging Demo**, choose
**Uninstall**, and follow the prompt. The uninstaller removes:

- the application and Start-menu shortcuts;
- the downloaded MAEST model under `%LOCALAPPDATA%\MusicClassification`;
- the application's Hugging Face cache.

Audio selected from another folder is not removed. This cleanup behavior is
declared in `packaging/windows/installer.iss`.

For a source-code checkout rather than the installer, close the program and
delete `.venv-demo`, `models/mtg-upf-maest-519l`, and `.cache` to remove the
runtime while keeping the source. Deleting the whole checkout removes
everything created inside it, including any personal files you added there;
back those up first. The documented virtual-environment setup does not install
the demo's dependencies system-wide or remove your existing Python installation.

## Verify the download (optional)

In PowerShell, run the following from the folder containing the downloaded
installer (replace `<version>` with the release version):

```powershell
Get-FileHash .\MusicClassification-Setup-<version>-win64.exe -Algorithm SHA256
```

Compare the displayed hash with `SHA256SUMS.txt` from the same release. This
detects changed or incomplete downloads; it does not authenticate a publisher.

## Build and release (maintainer)

The `Build Windows installer` workflow runs on a real Windows x64 runner. It
runs the unit tests, creates the PyInstaller bundle, downloads and verifies the
pinned upstream model, performs end-to-end inference on the included example,
creates the Inno Setup installer, and finishes with a silent install/uninstall
test that checks both program files and downloaded model/cache cleanup.

Run the workflow manually to obtain a temporary GitHub Actions artifact. To
publish a permanent GitHub Release, push a version tag:

```bash
git tag v1.0.0
git push origin v1.0.0
```

The workflow publishes the installer, its SHA-256 checksum, and the reviewer
instructions in `docs/RELEASE_NOTES_WINDOWS.md`. The fourth source commit adds
the first desktop release, `v1.0.0`; source revision numbers and app versions
are separate. PyInstaller is
not a cross-compiler, so a Windows application must be built on Windows; the
workflow supplies that environment rather than attempting to cross-build from
macOS.

## Signing status

This academic portfolio release intentionally does not use Authenticode code
signing. The installer and application are unsigned. Windows may display an
unknown-publisher or SmartScreen warning; institution-managed computers may
prevent installation under their security policies. The project does not
require or recommend disabling antivirus, SmartScreen, or institutional controls.
Reviewers who cannot install it can read the project results and figures on
the repository homepage. A checksum is not a substitute for publisher
authentication.
