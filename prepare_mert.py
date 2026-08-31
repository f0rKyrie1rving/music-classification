"""Download the pinned public encoder; verify bytes without importing its code."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "models/mert-v0-public"
REVISION = "e8413e398b3180ca488534aea68fd829402044a4"
REPO = "https://huggingface.co/m-a-p/MERT-v0-public"
FILES = {
    "README.md": (6703, "cbb12c02a7019dd5fe20430f1bd460c4ce4d6dcb"),
    "config.json": (1798, "7f3a56153c22ce1ebb3af17fbc7bc23b8dd7ff21"),
    "configuration_MERT.py": (4890, "c8d1359e14f2cd567db5d5079960b3590892d531"),
    "modeling_MERT.py": (18033, "e7ac06ba5bd4a8e633ce6a42e230e8935322c7ee"),
    "preprocessor_config.json": (212, "39b6440871e1ee5445df0885a53e18561b8e15d6"),
    "pytorch_model.bin": (377552987, "9b25bde740483579d9895f35d074a949f6593ef48449b6d76e26ee3c0e5e9acb"),
}


def verify(path, expected):
    data = path.read_bytes()
    size, reference = expected
    if len(data) != size:
        raise ValueError(f"Size mismatch: {path.name}")
    sha256 = hashlib.sha256(data).hexdigest()
    # Small-file IDs from the official API are Git blob SHA-1s, not raw SHA-1s.
    actual = sha256 if len(reference) == 64 else hashlib.sha1(
        f"blob {len(data)}\0".encode() + data).hexdigest()
    if actual != reference:
        raise ValueError(f"Checksum mismatch: {path.name}")
    return sha256


def main(proxy=None):
    DEST.mkdir(parents=True, exist_ok=True)
    verified = {}
    for name, expected in FILES.items():
        target = DEST / name
        if not target.exists():
            part = target.with_name(name + ".part")
            command = ["/usr/bin/curl", "--fail", "--location", "--silent", "--show-error",
                       "--connect-timeout", "20", "--max-time", "900", "--retry", "2",
                       "--continue-at", "-", "--output", str(part)]
            if proxy:
                command += ["--proxy", proxy]
            command.append(f"{REPO}/resolve/{REVISION}/{name}")
            print(f"Downloading {name}: {expected[0]:,} bytes", flush=True)
            subprocess.run(command, check=True)
            verify(part, expected)
            part.replace(target)
        verified[name] = verify(target, expected)
        print(f"Verified {name}", flush=True)
    sources = dict(model_id="m-a-p/MERT-v0-public", revision=REVISION,
        metadata_source=f"https://huggingface.co/api/models/m-a-p/MERT-v0-public/revision/{REVISION}?blobs=true",
        license="cc-by-nc-4.0", files_sha256=verified,
        upstream_checks={k: dict(size=v[0], digest=v[1], algorithm="sha256" if len(v[1]) == 64 else "git-blob-sha1") for k, v in FILES.items()},
        code_executed_by_downloader=False, pretraining_overlap="not audited")
    (DEST / "SOURCES.json").write_text(json.dumps(sources, indent=2) + "\n")
    print("Download verified. Custom source must be reviewed before model loading.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", help="Optional existing HTTP proxy; no system settings are changed.")
    main(parser.parse_args().proxy)
