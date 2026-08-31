"""Pilot and resumable development-only extraction for the frozen MERT experiment."""

import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from improve_model import OUT, PLAN, load_development
from mert_features import DEST, ROOT, MertEncoder, sha256


def metadata(device):
    return dict(encoder_manifest_sha256=sha256(DEST / "SOURCES.json"),
        extractor_source_sha256=sha256(ROOT / "mert_features.py"),
        driver_source_sha256=sha256(Path(__file__)), plan_sha256=sha256(PLAN), device=device,
        versions=json.dumps({p: version(p) for p in ("torch", "transformers", "librosa", "numpy", "soxr")}, sort_keys=True))


def pilot(device):
    OUT.mkdir(parents=True, exist_ok=True)
    audio = ROOT / "data/previews/track_0913702_30s.wav"
    before = time.perf_counter()
    encoder = MertEncoder(device)
    load_seconds = time.perf_counter() - before
    timings, vectors = [], []
    for _ in range(2):
        before = time.perf_counter()
        vectors.append(encoder.extract(audio))
        timings.append(time.perf_counter() - before)
    np.testing.assert_allclose(vectors[0], vectors[1], rtol=1e-5, atol=1e-6)
    result = dict(audio=str(audio.relative_to(ROOT)), audio_sha256=sha256(audio), shape=list(vectors[0].shape),
        load_seconds=load_seconds, extraction_seconds=timings, finite=bool(np.isfinite(vectors).all()),
        max_abs_repeat_difference=float(np.max(np.abs(vectors[0]-vectors[1]))),
        python=platform.python_version(), platform=platform.platform(),
        encoder_training=encoder.model.training, parameters_require_grad=any(p.requires_grad for p in encoder.model.parameters()),
        mps_available=encoder.torch.backends.mps.is_available(), **metadata(device))
    np.save(OUT / f"mert_pilot_{device}.npy", vectors[0], allow_pickle=False)
    (OUT / f"mert_pilot_{device}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def extract(device):
    plan, rows, _, _ = load_development()
    meta = metadata(device)
    pilot_path = OUT / f"mert_pilot_{device}.json"
    checked_pilot = json.loads(pilot_path.read_text())
    if any(checked_pilot[k] != v for k, v in meta.items()):
        raise ValueError("Run the pilot for this exact extractor/runtime first.")
    ids = np.array([r["track_id"] for r in rows])
    hashes = np.array([r["wav_sha256"] for r in rows])
    final, partial = OUT / "mert_features.npz", OUT / "mert_partial.npz"
    cache = final if final.exists() else partial
    vectors = []
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            n = len(data["ids"])
            if any(str(data[k]) != str(v) for k, v in meta.items()) or n > len(rows):
                raise ValueError("Stale encoder cache: do not mix protocols or runtimes.")
            if not np.array_equal(data["ids"], ids[:n]) or not np.array_equal(data["audio_hashes"], hashes[:n]):
                raise ValueError("Stale encoder audio/ID cache.")
            if data["x"].shape != (n, 768) or not np.isfinite(data["x"]).all():
                raise ValueError("Invalid partial encoder matrix.")
            vectors = list(data["x"])
    if len(vectors) == len(rows):
        print("Complete development cache verified; no new inference required.")
        return
    # Freeze execution details before the first development feature is computed.
    execution_plan = ROOT / "experiments/mert_execution_plan.json"
    if execution_plan.exists() and json.loads(execution_plan.read_text()) != meta:
        raise ValueError("MERT execution already frozen with different settings.")
    execution_plan.write_text(json.dumps(meta, indent=2) + "\n")
    encoder = MertEncoder(device)
    begin, resumed = time.perf_counter(), len(vectors)
    for row in rows[resumed:]:
        vectors.append(encoder.extract(ROOT / row["audio_file"]))
        n = len(vectors)
        if n % 10 == 0 or n == len(rows):
            temp = OUT / "mert_writing.npz"
            np.savez_compressed(temp, x=np.array(vectors), ids=ids[:n], audio_hashes=hashes[:n], **meta)
            temp.replace(final if n == len(rows) else partial)
            print(f"MERT development tracks: {n}/{len(rows)}; {time.perf_counter()-begin:.1f}s this run", flush=True)
    load_development()
    (OUT / "mert_extraction_run.json").write_text(json.dumps(dict(
        total_tracks=len(rows), resumed_tracks=resumed, new_tracks=len(rows)-resumed,
        extraction_seconds=time.perf_counter()-begin, test_tracks=0, cache_sha256=sha256(final), **meta), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["pilot", "extract"])
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    (pilot if args.action == "pilot" else extract)(args.device)
