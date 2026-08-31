"""Download the pinned official Discogs-MAEST model and record its digest."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEST = ROOT / "models/discogs-maest-30s-pw-519l"
MODEL_NAME = "discogs-maest-30s-pw-519l-2.pb"
METADATA_NAME = "discogs-maest-30s-pw-519l-2.json"
BASE_URL = "https://essentia.upf.edu/models/feature-extractors/maest"
EXPECTED_SIZES = {MODEL_NAME: 347_996_675, METADATA_NAME: 23_232}


def sha256(path):
    block, digest = 1024 * 1024, hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def validate_metadata(path):
    data = json.loads(Path(path).read_text())
    outputs = {item["name"]: item for item in data["schema"]["outputs"]}
    expected_link = f"{BASE_URL}/{MODEL_NAME}"
    if (data.get("name") != "MAEST" or data.get("version") != "2"
            or data.get("framework") != "tensorflow" or data.get("link") != expected_link
            or len(data.get("classes", [])) != 519
            or data.get("dataset", {}).get("size") != "4M full tracks"
            or data.get("schema", {}).get("inputs", [{}])[0].get("shape") != [1, 1876, 96]
            or outputs.get("PartitionedCall/Identity_7", {}).get("shape", [])[-1:] != [768]):
        raise ValueError("Official MAEST metadata differs from the frozen candidate.")
    return data


def verify_file(path, expected_size):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError(f"Size mismatch for {path.name}: expected {expected_size:,} bytes.")
    return sha256(path)


def main(proxy=None):
    DEST.mkdir(parents=True, exist_ok=True)
    observed = {}
    for name in (METADATA_NAME, MODEL_NAME):
        target = DEST / name
        if not target.exists():
            part = target.with_name(name + ".part")
            command = ["/usr/bin/curl", "--fail", "--location", "--silent", "--show-error",
                       "--connect-timeout", "20", "--max-time", "1800", "--retry", "2",
                       "--continue-at", "-", "--output", str(part)]
            if proxy:
                command += ["--proxy", proxy]
            command.append(f"{BASE_URL}/{name}")
            print(f"Downloading {name}: {EXPECTED_SIZES[name]:,} bytes", flush=True)
            subprocess.run(command, check=True)
            verify_file(part, EXPECTED_SIZES[name])
            if name == METADATA_NAME:
                validate_metadata(part)
            part.replace(target)
        observed[name] = verify_file(target, EXPECTED_SIZES[name])
        if name == METADATA_NAME:
            validate_metadata(target)
        print(f"Verified {name}: {observed[name]}", flush=True)
    receipt = {
        "model": "discogs-maest-30s-pw-519l",
        "version": "2",
        "release_date": "2025-01-22",
        "official_base_url": BASE_URL,
        "license": "CC BY-NC-ND 4.0 for non-commercial use",
        "files": {name: {"url": f"{BASE_URL}/{name}",
                         "expected_size_from_official_index": EXPECTED_SIZES[name],
                         "observed_sha256": observed[name]}
                  for name in (METADATA_NAME, MODEL_NAME)},
        "upstream_checksum_available": False,
        "checksum_note": "Official index publishes sizes but no checksum; SHA-256 pins the locally received official files.",
        "pretraining_overlap_with_jamendo": "not auditable from the unreleased Discogs23 metadata",
    }
    (DEST / "SOURCES.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("MAEST official files verified and source receipt written.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", help="Optional existing HTTP proxy; system settings are unchanged.")
    main(parser.parse_args().proxy)
