"""Download the pinned MERT-v1 encoder and verify it without executing code."""

import argparse
import json
import subprocess
from pathlib import Path

from prepare_mert import verify


ROOT = Path(__file__).resolve().parent
DEST = ROOT / "models/mert-v1-95m"
REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
REPO = "https://huggingface.co/m-a-p/MERT-v1-95M"
FILES = {
    "README.md": (6756, "ff747b4410bf0dd7bca09ae5b765acf60c489127"),
    "config.json": (1817, "8061aff2203f5caeb5c58628dba945de3ef71079"),
    "configuration_MERT.py": (5340, "91f6098a642bb1e6498beea92025e49fa619e49d"),
    "modeling_MERT.py": (18033, "e7ac06ba5bd4a8e633ce6a42e230e8935322c7ee"),
    "preprocessor_config.json": (211, "12d419697ffa30069405b3ace413be9a997e9f7f"),
    "pytorch_model.bin": (377552987, "a2b8b747f72c06e0595aeae41ae5473f4364938c6b39b2c58be38c48e6bd3fcd"),
}


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
    sources = {
        "model_id": "m-a-p/MERT-v1-95M", "revision": REVISION,
        "metadata_source": f"https://huggingface.co/api/models/m-a-p/MERT-v1-95M/tree/{REVISION}?recursive=true&expand=true",
        "license": "cc-by-nc-4.0", "files_sha256": verified,
        "upstream_checks": {name: {"size": expected[0], "digest": expected[1],
                            "algorithm": "sha256" if len(expected[1]) == 64 else "git-blob-sha1"}
                            for name, expected in FILES.items()},
        "excluded_upstream_file": "MERT-v1-95M_fairseq.pt",
        "excluded_reason": "Not needed for Transformers inference and has broader pickle imports.",
        "code_executed_by_downloader": False,
        "pretraining_overlap": "not audited",
    }
    (DEST / "SOURCES.json").write_text(json.dumps(sources, indent=2) + "\n")
    print("MERT-v1 download verified; local custom code review is still required.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", help="Optional existing HTTP proxy; system settings are unchanged.")
    main(parser.parse_args().proxy)
