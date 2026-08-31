"""Acquire a small, licensed MTG-Jamendo subset without downloading whole TARs.

Preparation utility, separate from the core classifier. Commands: index, plan,
download. Partial MP3 prefixes are NEVER reported as full-file hash verified.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
import time

import numpy as np
import soundfile as sf
import soxr

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/source"
POOL_PATH = SOURCE / "training_pool.json"
MANIFEST = ROOT / "data/dataset_manifest.json"
CHUNK = 65536
SEED = "music-portfolio-2026-08-27-v1"


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def get_range(url, start, size):
    """Require an exact HTTP 206 range and bound every network response."""
    last_error = None
    for attempt in range(3):
        with tempfile.TemporaryDirectory(prefix="music-download-") as tmp:
            body, headers = Path(tmp) / "body", Path(tmp) / "headers"
            result = subprocess.run([
                "/usr/bin/curl", "--fail", "--location", "--silent", "--show-error",
                "--connect-timeout", "8", "--max-time", "25", "--range",
                f"{start}-{start + size - 1}", "--max-filesize", str(size),
                "--dump-header", str(headers), "--output", str(body), url,
            ], capture_output=True, text=True)
            if result.returncode == 0:
                match = re.search(r"content-range:\s*bytes (\d+)-(\d+)/(\d+)",
                                  headers.read_text(), re.I)
                data = body.read_bytes()
                if match and tuple(map(int, match.groups()[:2])) == (start, start + size - 1) and len(data) == size:
                    return data, int(match[3])
                last_error = "Server did not return the exact requested byte range."
            else:
                last_error = result.stderr.strip()
        time.sleep(attempt + 1)
    raise RuntimeError(f"Range download failed: {last_error}")


def index_archive(shard, max_headers):
    path = SOURCE / f"archive_index_{shard}.json"
    pool = json.loads(POOL_PATH.read_text())
    wanted = {r["archive_member"] for r in pool["records"]}
    url = f"https://cdn.freesound.org/mtg-jamendo/raw_30s/audio-low/raw_30s_audio-low-{shard}.tar"
    state = json.loads(path.read_text()) if path.exists() else dict(
        archive_url=url, next_offset=0, headers_scanned=0, members={}, complete=False)
    while state["headers_scanned"] < max_headers and not state["complete"]:
        offset = state["next_offset"]
        data, total = get_range(url, offset, 512)
        if not data.strip(b"\0"):
            state["complete"] = True
            break
        member = tarfile.TarInfo.frombuf(data, "utf-8", "strict")
        if member.isfile() and member.name in wanted:
            state["members"][member.name] = dict(offset=offset + 512, size=member.size)
        state.update(next_offset=offset + 512 + ((member.size + 511) // 512) * 512,
                     archive_bytes=total, headers_scanned=state["headers_scanned"] + 1)
        if state["headers_scanned"] % 25 == 0:
            save_json(path, state)
        if state["headers_scanned"] % 100 == 0:
            print(f'Index {shard}: {state["headers_scanned"]} headers, {len(state["members"])} eligible tracks', flush=True)
    save_json(path, state)
    return {"archive": shard, "headers": state["headers_scanned"], "eligible": len(state["members"])}


def choose_split(rows, labels, count, minimum, artist_cap):
    """Deterministic coverage sampling, with explicit artist and background caps."""
    ordered = sorted(rows, key=lambda r: hashlib.sha256((SEED + r["track_id"]).encode()).hexdigest())
    selected, used, artists, positives = [], set(), Counter(), Counter()
    def add(row):
        selected.append(row)
        used.add(row["track_id"])
        artists[row["artist_id"]] += 1
        positives.update(set(row["tags"]) & set(labels))
    def available(row):
        return row["track_id"] not in used and artists[row["artist_id"]] < artist_cap
    for row in ordered:
        if not set(row["tags"]) & set(labels) and available(row) and len(selected) < round(count * 0.1):
            add(row)
    while min(positives[label] for label in labels) < minimum:
        label = min(labels, key=lambda tag: positives[tag])
        choices = [r for r in ordered if available(r) and label in r["tags"]]
        if not choices:
            raise ValueError(f"Insufficient artist-diverse coverage for {label}; extend the index.")
        row = max(choices, key=lambda r: sum(positives[t] < minimum for t in labels if t in r["tags"]))
        add(row)
    for row in ordered:
        if len(selected) >= count:
            break
        if available(row) and set(row["tags"]) & set(labels):
            add(row)
    if len(selected) != count:
        raise ValueError(f"Need {count} tracks, found {len(selected)}; extend the index.")
    return selected


def make_plan():
    if MANIFEST.exists():
        raise FileExistsError("A frozen manifest already exists; do not silently resample it.")
    pool = json.loads(POOL_PATH.read_text())
    indexed = {}
    for shard in pool["candidate_archives"]:
        index = json.loads((SOURCE / f"archive_index_{shard}.json").read_text())
        for name, entry in index["members"].items():
            indexed[name] = dict(**entry, archive_url=index["archive_url"], archive_bytes=index["archive_bytes"])
    eligible = [dict(r, archive=indexed[r["archive_member"]]) for r in pool["records"]
                if r["archive_member"] in indexed and r["track_id"] not in
                {"track_0543400", "track_0207501", "track_0913702"}]
    selected = []
    for split, count, minimum, cap in [("train", 300, 50, 3), ("validation", 90, 15, 2), ("test", 90, 15, 2)]:
        selected.extend(choose_split([r for r in eligible if r["split"] == split], pool["labels"], count, minimum, cap))
    for row in selected:
        row["audio_file"] = f'data/training_audio/{row["track_id"]}.wav'
        row["targets"] = [int(tag in row["tags"]) for tag in pool["labels"]]
    manifest = dict(dataset=pool["dataset"], official_split="split-0", labels=pool["labels"],
                    source_blobs=pool["source_blobs"], metadata_license=pool["metadata_license"],
                    selection_seed=SEED, sampling="Coverage-enriched subset from archives 00–03; 10% background; not the official benchmark distribution.",
                    audio_policy="Non-ND Creative Commons audio, used only for non-commercial research; individual licenses retained.",
                    excerpt=dict(offset_seconds=0, duration_seconds=30, sample_rate=22050, channels=1, subtype="PCM_16"),
                    tracks=selected)
    save_json(MANIFEST, manifest)
    print(f"Frozen {len(selected)} tracks in {MANIFEST}", flush=True)


def leading_id3_size(data):
    """Return the leading ID3v2 tag's total length, including header/footer."""
    if data[:3] != b"ID3":
        return 0
    if len(data) < 10 or data[3] not in (2, 3, 4) or any(b & 128 for b in data[6:10]):
        raise ValueError("Invalid or unsupported ID3v2 header.")
    size = sum(b << (7 * (3 - i)) for i, b in enumerate(data[6:10]))
    return 10 + size + (10 if data[3] == 4 and data[5] & 0x10 else 0)


def extend_part(path, content, relative_start, target, archive):
    while len(content) < target:
        block, total = get_range(archive["archive_url"], archive["offset"] + relative_start + len(content),
                                 min(CHUNK, target - len(content)))
        if total != archive["archive_bytes"]:
            raise ValueError("Archive size changed since indexing.")
        with path.open("ab") as file:
            file.write(block)
        content += block
    return content


def download_track(row):
    dest = ROOT / row["audio_file"]
    receipts = SOURCE / "download_receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts / f'{row["track_id"]}.json'
    if dest.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if hashlib.sha256(dest.read_bytes()).hexdigest() == receipt["wav_sha256"]:
            return receipt
    prefix_dir = SOURCE / "audio_prefixes"
    prefix_dir.mkdir(exist_ok=True)
    prefix_path = prefix_dir / f'{row["track_id"]}.mp3.part'
    prefix = prefix_path.read_bytes() if prefix_path.exists() else b""
    archive = row["archive"]
    # Read the header first: a multi-megabyte album cover need not be fetched.
    prefix = extend_part(prefix_path, prefix, 0, min(CHUNK, archive["size"]), archive)
    tag_end = leading_id3_size(prefix)
    if tag_end >= archive["size"]:
        raise ValueError("ID3 tag leaves no audio in the indexed member.")
    skip_tag = tag_end > 6 * CHUNK and tag_end >= len(prefix)
    if skip_tag:
        encoded_path = prefix_dir / f'{row["track_id"]}.audio.mp3.part'
        encoded = encoded_path.read_bytes() if encoded_path.exists() else b""
        relative_start = tag_end
        maximum = min(archive["size"] - tag_end, 20 * CHUNK - len(prefix))
    else:
        encoded_path, encoded, relative_start = prefix_path, prefix, 0
        maximum = min(archive["size"], 20 * CHUNK)
    if maximum <= 0 or len(encoded) > maximum:
        raise ValueError("Cached bytes exceed the per-track acquisition budget.")
    target = min(maximum, max(6 * CHUNK, len(encoded)))
    while True:
        encoded = extend_part(encoded_path, encoded, relative_start, target, archive)
        try:
            with sf.SoundFile(io.BytesIO(encoded)) as audio:
                rate = audio.samplerate
                samples = audio.read(frames=30*rate, dtype="float64", always_2d=True)
        except RuntimeError:
            # Some MP3s begin with large embedded images or metadata tags.
            if target >= maximum:
                raise
            target = min(maximum, target + 2 * CHUNK)
            continue
        if len(samples) == 30 * rate:
            break
        if target >= maximum:
            raise ValueError(f'Not enough decodable audio within bounded prefix: {row["track_id"]}')
        target = min(maximum, target + 2 * CHUNK)
    mono = samples.mean(axis=1)
    y = soxr.resample(mono, rate, 22050, quality="HQ") if rate != 22050 else mono
    if len(y) != 661500 or not np.isfinite(y).all() or np.sqrt(np.mean(y**2)) < 1e-5:
        raise ValueError(f'Invalid or near-silent excerpt: {row["track_id"]}')
    if not skip_tag:
        prefix = encoded
    full_hash_verified = not skip_tag and len(prefix) == archive["size"]
    prefix_hash = hashlib.sha256(prefix).hexdigest()
    if full_hash_verified and prefix_hash != row["expected_full_sha256"]:
        raise ValueError("Full MP3 checksum mismatch.")
    dest.parent.mkdir(exist_ok=True)
    temporary = dest.with_suffix(".tmp.wav")
    sf.write(temporary, y, 22050, subtype="PCM_16")
    temporary.replace(dest)
    byte_ranges = [dict(byte_range=[archive["offset"], archive["offset"] + len(prefix) - 1],
                       sha256=prefix_hash, cache_file=str(prefix_path.relative_to(ROOT)))]
    if skip_tag:
        start = archive["offset"] + relative_start
        byte_ranges.append(dict(byte_range=[start, start + len(encoded) - 1],
                                sha256=hashlib.sha256(encoded).hexdigest(),
                                cache_file=str(encoded_path.relative_to(ROOT))))
    receipt = dict(track_id=row["track_id"], downloaded_bytes=len(prefix) + (len(encoded) if skip_tag else 0),
                    byte_ranges=byte_ranges, leading_id3_bytes=tag_end,
                    skipped_id3_bytes=tag_end - len(prefix) if skip_tag else 0,
                    prefix_sha256=prefix_hash, full_mp3_sha256_verified=full_hash_verified,
                    wav_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                    frames=661500, sample_rate=22050, original_sample_rate=rate,
                    retrieved_at_utc=datetime.now(timezone.utc).isoformat())
    save_json(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["index", "plan", "download"])
    parser.add_argument("--max-headers", type=int, default=320)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "index":
        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(lambda s: index_archive(s, args.max_headers), ["00", "01", "02", "03"]):
                print(result, flush=True)
    elif args.command == "plan":
        make_plan()
    else:
        rows = json.loads(MANIFEST.read_text())["tracks"]
        if args.limit:
            rows = rows[:args.limit]
        failures, completed = [], []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(download_track, row): row for row in rows}
            for future in as_completed(jobs):
                row = jobs[future]
                try:
                    completed.append(future.result())
                except Exception as error:
                    failures.append(dict(track_id=row["track_id"], error=str(error)))
                    print(f'FAILED {row["track_id"]}: {error}', flush=True)
                if (len(completed)+len(failures)) % 10 == 0:
                    print(f'Downloaded {len(completed)}/{len(rows)}; failures={len(failures)}', flush=True)
        save_json(SOURCE / "download_status.json", dict(completed=len(completed), requested=len(rows), failures=failures))
        if failures:
            raise SystemExit("Incomplete download: inspect data/source/download_status.json and retry.")
        print(f"Verified {len(completed)} decoded 30-second excerpts.", flush=True)


if __name__ == "__main__":
    main()
