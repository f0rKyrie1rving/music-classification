"""Download the pinned official MTG-UPF MAEST safetensors repository."""

import argparse
import json
import subprocess
from pathlib import Path

from prepare_mert import verify


ROOT = Path(__file__).resolve().parent
DEST = ROOT / "models/mtg-upf-maest-519l"
MODEL_ID = "mtg-upf/discogs-maest-30s-pw-129e-519l"
REVISION = "6c35f32a350f74351870937d5ae0bae1d898d1df"
REPO = f"https://huggingface.co/{MODEL_ID}"
FILES = {
    "README.md": (5174, "bc5f30d6632ac0efdc7be2e9095e9e9579af2e33"),
    "config.json": (36729, "7e67242387d27954e28fce3120f2ce8e68bcf85b"),
    "feature_extraction_maest.py": (10162, "593f9332598434b8d1f109c48fb8e52abe03ec38"),
    "preprocessor_config.json": (476, "39b8883f8bb1322f383ef24ae4dadee628fbdf63"),
    "model.safetensors": (347827260, "881cec6abdcb6ef986367e6c0db02dc760cc0f88a52b625cb9d6e2b54d505548"),
}


def main(proxy=None, metadata_only=False):
    DEST.mkdir(parents=True, exist_ok=True)
    names = [name for name in FILES if not metadata_only or name != "model.safetensors"]
    verified = {}
    for name in names:
        target = DEST / name
        if not target.exists():
            part = target.with_name(name + ".part")
            command = ["/usr/bin/curl", "--fail", "--location", "--silent", "--show-error",
                       "--connect-timeout", "20", "--max-time", "1800", "--retry", "2",
                       "--continue-at", "-", "--output", str(part)]
            if proxy:
                command += ["--proxy", proxy]
            command.append(f"{REPO}/resolve/{REVISION}/{name}")
            print(f"Downloading {name}: {FILES[name][0]:,} bytes", flush=True)
            subprocess.run(command, check=True)
            verify(part, FILES[name])
            part.replace(target)
        verified[name] = verify(target, FILES[name])
        print(f"Verified {name}: {verified[name]}", flush=True)
    if metadata_only:
        print("MAEST metadata files verified; weights deliberately not requested.", flush=True)
        return
    receipt = {
        "model_id": MODEL_ID, "revision": REVISION,
        "official_organization": "Music Technology Group (Universitat Pompeu Fabra)",
        "metadata_source": f"https://huggingface.co/api/models/{MODEL_ID}/tree/{REVISION}?recursive=true&expand=true",
        "model_format": "safetensors", "files_sha256": verified,
        "upstream_checks": {name: {"size": expected[0], "digest": expected[1],
                            "algorithm": "sha256" if len(expected[1]) == 64 else "git-blob-sha1"}
                            for name, expected in FILES.items()},
        "model_license": "not declared in the Hugging Face card; treated conservatively as CC BY-NC-ND 4.0 per Essentia model licensing",
        "feature_extractor_source_license": "Apache-2.0 as declared in the pinned source header",
        "pretraining_overlap_with_jamendo": "not auditable from public metadata",
        "remote_code_disabled_after_download": True,
    }
    (DEST / "SOURCES.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Official MTG-UPF MAEST repository verified; review local feature code before use.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", help="Optional existing HTTP proxy; system settings are unchanged.")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    main(args.proxy, args.metadata_only)
