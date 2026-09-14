"""Runtime paths and verified model download for the desktop application."""

from __future__ import annotations

import json
import os
import platform
import ssl
import sys
import time
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi
import prepare_maest_hf
from prepare_maest_hf import FILES, MODEL_ID, REPO, REVISION
from prepare_mert import verify


APP_DIRECTORY = "MusicClassification"
DATA_DIRECTORY_ENV = "MUSIC_CLASSIFICATION_DATA_DIR"
CHUNK_SIZE = 1024 * 1024
ProgressCallback = Callable[[str, float | None], None]
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def default_user_data_dir(system=None, environ=None, home=None):
    """Return a per-user writable directory without changing the filesystem."""
    environ = os.environ if environ is None else environ
    override = environ.get(DATA_DIRECTORY_ENV)
    if override:
        return Path(override).expanduser()

    system = platform.system() if system is None else system
    home = Path.home() if home is None else Path(home)
    if system == "Windows":
        base = environ.get("LOCALAPPDATA")
        return (Path(base) if base else home / "AppData/Local") / APP_DIRECTORY
    if system == "Darwin":
        return home / "Library/Application Support" / APP_DIRECTORY
    base = environ.get("XDG_DATA_HOME")
    return (Path(base) if base else home / ".local/share") / APP_DIRECTORY


def configure_runtime(data_root=None):
    """Point desktop model and cache writes at a user-writable location."""
    root = Path(data_root) if data_root is not None else default_user_data_dir()
    model_dir = root / "models/mtg-upf-maest-519l"
    cache_dir = root / ".cache/huggingface"
    root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    # The research modules keep repository-local defaults. The desktop entry
    # point redirects only its own process before importing the encoder.
    prepare_maest_hf.DEST = model_dir
    loaded_encoder = sys.modules.get("maest_hf_features")
    if loaded_encoder is not None:
        loaded_encoder.DEST = model_dir
    return root, model_dir


def model_files_present(model_dir):
    """Cheap startup check; the encoder performs full digest verification."""
    model_dir = Path(model_dir)
    receipt = model_dir / "SOURCES.json"
    if not receipt.is_file():
        return False
    try:
        metadata = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if metadata.get("model_id") != MODEL_ID or metadata.get("revision") != REVISION:
        return False
    return all(
        (model_dir / name).is_file() and (model_dir / name).stat().st_size == expected[0]
        for name, expected in FILES.items()
    )


def _report(callback, message, fraction=None):
    if callback is not None:
        callback(message, fraction)


def _download_file(url, target, expected, callback=None, attempts=3):
    """Download one file with bounded retries and resumable partial data."""
    target = Path(target)
    part = target.with_name(target.name + ".part")
    expected_size = expected[0]
    target.parent.mkdir(parents=True, exist_ok=True)

    if part.is_file() and part.stat().st_size == expected_size:
        try:
            verify(part, expected)
        except ValueError:
            part.unlink()
        else:
            part.replace(target)
            return
    if part.is_file() and part.stat().st_size > expected_size:
        part.unlink()

    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            offset = part.stat().st_size if part.is_file() else 0
            headers = {"User-Agent": "MusicClassificationDesktop/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = Request(url, headers=headers)
            with urlopen(request, timeout=60, context=TLS_CONTEXT) as response:
                status = getattr(response, "status", response.getcode())
                if offset and status != 206:
                    offset = 0
                    mode = "wb"
                else:
                    mode = "ab" if offset else "wb"
                downloaded = offset
                with part.open(mode) as stream:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        stream.write(chunk)
                        downloaded += len(chunk)
                        fraction = min(downloaded / expected_size, 1.0)
                        _report(callback, f"Downloading {target.name}", fraction)
            if part.stat().st_size != expected_size:
                raise ValueError(
                    f"Incomplete download for {target.name}: "
                    f"{part.stat().st_size:,} of {expected_size:,} bytes"
                )
            verify(part, expected)
            part.replace(target)
            return
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            last_error = error
            if attempt < attempts:
                _report(callback, f"Retrying {target.name} ({attempt}/{attempts})")
                time.sleep(attempt)
    raise RuntimeError(f"Could not download {target.name}: {last_error}") from last_error


def prepare_desktop_model(data_root=None, callback=None):
    """Retrieve the pinned MAEST files and write the audited source receipt."""
    _, model_dir = configure_runtime(data_root)
    verified = {}
    for name, expected in FILES.items():
        target = model_dir / name
        valid = False
        if target.is_file():
            try:
                verified[name] = verify(target, expected)
                valid = True
            except ValueError:
                target.unlink()
        if not valid:
            _report(callback, f"Preparing {name}")
            _download_file(
                f"{REPO}/resolve/{REVISION}/{name}", target, expected, callback
            )
            verified[name] = verify(target, expected)
        _report(callback, f"Verified {name}")

    receipt = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "official_organization": "Music Technology Group (Universitat Pompeu Fabra)",
        "metadata_source": (
            f"https://huggingface.co/api/models/{MODEL_ID}/tree/"
            f"{REVISION}?recursive=true&expand=true"
        ),
        "model_format": "safetensors",
        "files_sha256": verified,
        "upstream_checks": {
            name: {
                "size": expected[0],
                "digest": expected[1],
                "algorithm": "sha256" if len(expected[1]) == 64 else "git-blob-sha1",
            }
            for name, expected in FILES.items()
        },
        "model_license": (
            "not declared in the Hugging Face card; treated conservatively as "
            "CC BY-NC-ND 4.0 per Essentia model licensing"
        ),
        "feature_extractor_source_license": (
            "Apache-2.0 as declared in the pinned source header"
        ),
        "pretraining_overlap_with_jamendo": "not auditable from public metadata",
        "remote_code_disabled_after_download": True,
    }
    receipt_path = model_dir / "SOURCES.json"
    temporary = receipt_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    temporary.replace(receipt_path)
    _report(callback, "Model ready", 1.0)
    return model_dir
