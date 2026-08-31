"""Acquire and verify only the frozen 239-track final holdout."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from mert_features import sha256
from prepare_dataset import download_track, save_json
from prepare_expansion import MANIFEST, ROOT, load_frozen
from repair_short_audio import repair


PLAN = ROOT / "experiments/final_holdout_plan.json"
PROTOCOL = ROOT / "experiments/FINAL_HOLDOUT_PROTOCOL.md"
SOURCE = ROOT / "data/source"
STATUS = SOURCE / "final_holdout_download_status.json"
AUDIT = ROOT / "data/final_holdout_audit.json"


def holdout_rows():
    plan = json.loads(PLAN.read_text())
    if (plan["protocol_sha256"] != sha256(PROTOCOL)
            or plan["expanded_manifest_sha256"] != sha256(MANIFEST)
            or plan["preparer_source_sha256"] != sha256(Path(__file__))):
        raise ValueError("Final holdout acquisition inputs changed after freeze.")
    manifest = load_frozen()
    by_id = {row["track_id"]: row for row in manifest["tracks"]}
    rows = [by_id[track_id] for track_id in plan["holdout_ids"]]
    if (len(rows) != 239 or any(row["phase_role"] != "new_holdout" for row in rows)
            or set(plan["fit_ids"]) & set(plan["holdout_ids"])
            or set(plan["historical_test_ids"]) & set(plan["holdout_ids"])):
        raise ValueError("Frozen final holdout boundary is invalid.")
    return plan, rows


def acquire(workers):
    _, rows = holdout_rows(); failures, completed, repaired = [], [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(download_track, row): row for row in rows}
        for future in as_completed(jobs):
            row = jobs[future]
            try:
                completed.append(future.result())
            except Exception as error:
                if "Not enough decodable audio within bounded prefix" in str(error):
                    try:
                        completed.append(repair(row)); repaired.append(row["track_id"])
                    except Exception as repair_error:
                        failures.append({"track_id": row["track_id"],
                                         "error": str(error), "repair_error": str(repair_error)})
                else:
                    failures.append({"track_id": row["track_id"], "error": str(error)})
                if failures:
                    print(f"FAILED {row['track_id']}: {failures[-1]}", flush=True)
            done = len(completed) + len(failures)
            if done % 20 == 0 or done == len(rows):
                print(f"Holdout audio {len(completed)}/{len(rows)}; failures={len(failures)}",
                      flush=True)
    save_json(STATUS, {"requested": len(rows), "completed": len(completed),
                       "failures": failures, "short_source_repairs": repaired})
    if failures:
        raise SystemExit("Incomplete holdout acquisition; inspect the status file.")
    print(f"Acquired all {len(completed)} frozen holdout excerpts.")


def verify():
    plan, rows = holdout_rows(); records, rms_values, repairs = {}, [], []
    for row in rows:
        path = ROOT / row["audio_file"]
        receipt_path = SOURCE / "download_receipts" / f"{row['track_id']}.json"
        receipt = json.loads(receipt_path.read_text())
        digest = sha256(path)
        if receipt["track_id"] != row["track_id"] or receipt["wav_sha256"] != digest:
            raise ValueError(f"Holdout receipt mismatch: {row['track_id']}")
        info = sf.info(path)
        if (info.frames, info.samplerate, info.channels, info.subtype) != (661500, 22050, 1, "PCM_16"):
            raise ValueError(f"Unexpected holdout WAV format: {row['track_id']}")
        audio, _ = sf.read(path, dtype="float32")
        rms = float(np.sqrt(np.mean(audio.astype("float64") ** 2)))
        if not np.isfinite(audio).all() or rms < 1e-5:
            raise ValueError(f"Invalid holdout audio: {row['track_id']}")
        if "source_padding_rule" in receipt:
            repairs.append(row["track_id"])
        rms_values.append(rms)
        records[row["track_id"]] = {"wav_sha256": digest, "rms": rms,
                                     "receipt_sha256": sha256(receipt_path)}
    result = {"plan_sha256": sha256(PLAN), "manifest_sha256": sha256(MANIFEST),
              "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "phase_roles": dict(Counter(row["phase_role"] for row in rows)),
              "verified_tracks": len(rows), "historical_test_audio_used": 0,
              "min_rms": min(rms_values), "max_rms": max(rms_values),
              "short_source_repairs": repairs, "tracks": records}
    save_json(AUDIT, result)
    print(json.dumps({key: value for key, value in result.items() if key != "tracks"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "verify"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(); (acquire(args.workers) if args.command == "download" else verify())
