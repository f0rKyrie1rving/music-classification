"""Identity of the current desktop release, separate from the frozen v1.1 baseline."""

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
APP_VERSION = "1.2.0"
RELEASE_FILE = "artifacts/application_release.json"
RELEASE_SHA256 = "f1729a461f17c0b87a7668f8e189e35e4f5c5d2f5928ae8d631fc1a97b7d4a00"


def load_release(root=ROOT):
    """Reject a missing, changed or mismatched deployment receipt before inference."""
    root = Path(root).resolve()
    path = root / RELEASE_FILE
    if not path.is_file():
        raise FileNotFoundError("Application release receipt is missing; restore the application bundle.")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != RELEASE_SHA256:
        raise ValueError("Application release receipt checksum mismatch; restore the application bundle.")
    record = json.loads(content)
    if (record["format_version"] != 1 or record["application_version"] != APP_VERSION
            or record["validation"]["verdict"] != "go"):
        raise ValueError("Application release does not match the validated version.")
    directory = (root / record["candidate_directory"]).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Application model directory must be inside the application bundle.")
    for filename, key in (("candidate.json", "candidate_metadata_sha256"),
                          ("candidate.npz", "model_weights_sha256")):
        model_path = (directory / filename).resolve()
        if (model_path.parent != directory or not model_path.is_file()
                or hashlib.sha256(model_path.read_bytes()).hexdigest() != record[key]):
            raise ValueError(f"Validated application model missing or checksum mismatch: {filename}.")
    return record
