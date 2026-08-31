"""Pinned MERT-v1 layer features with local reviewed code and no network use."""

import contextlib
import importlib
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from prepare_mert import verify
from prepare_mert_v1 import DEST, FILES, REVISION


ROOT = Path(__file__).resolve().parent
RATE, CHUNK_SECONDS, CHUNKS, LAYERS, WIDTH = 24000, 5, 6, 13, 768
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache/numba"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def load_v1_chunks(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        with sf.SoundFile(path) as source:
            source_rate = source.samplerate
            samples = source.read(frames=30 * source_rate, dtype="float32", always_2d=True)
    except RuntimeError as error:
        raise ValueError(f"Invalid audio: {error}") from error
    if len(samples) != 30 * source_rate:
        raise ValueError("MERT-v1 input must contain at least 30 seconds.")
    audio = samples.mean(axis=1)
    if source_rate != RATE:
        audio = soxr.resample(audio, source_rate, RATE, quality="HQ")
    if len(audio) != RATE * CHUNK_SECONDS * CHUNKS:
        raise ValueError("MERT-v1 resampling did not produce exactly 30 seconds.")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples.")
    if np.sqrt(np.mean(audio.astype(np.float64) ** 2)) < 1e-5:
        raise ValueError("Audio is silent or below the RMS threshold (1e-5).")
    return audio.reshape(CHUNKS, RATE * CHUNK_SECONDS)


def summarize_v1_chunks(values):
    values = np.asarray(values)
    if values.shape != (CHUNKS, LAYERS, WIDTH) or not np.isfinite(values).all():
        raise ValueError(f"Expected finite {(CHUNKS, LAYERS, WIDTH)} layer chunks.")
    return values.mean(axis=0, dtype=np.float64)


class V1MertEncoder:
    def __init__(self, device="cpu"):
        import torch
        from transformers import Wav2Vec2FeatureExtractor

        self.torch, self.device = torch, device
        if device not in ("cpu", "mps") or (device == "mps" and not torch.backends.mps.is_available()):
            raise ValueError(f"Unavailable encoder device: {device}")
        sources = json.loads((DEST / "SOURCES.json").read_text())
        if sources["revision"] != REVISION or set(sources["files_sha256"]) != set(FILES):
            raise ValueError("MERT-v1 source receipt differs from the pinned version.")
        for name, expected in FILES.items():
            if verify(DEST / name, expected) != sources["files_sha256"][name]:
                raise ValueError(f"MERT-v1 source mismatch: {name}")
        package = types.ModuleType("portfolio_mert_v1")
        package.__path__ = [str(DEST)]
        sys.modules[package.__name__] = package
        with contextlib.redirect_stdout(sys.stderr):
            module = importlib.import_module("portfolio_mert_v1.modeling_MERT")
        config = module.MERTConfig(**json.loads((DEST / "config.json").read_text()))
        if (config.sample_rate, config.hidden_size, config.num_hidden_layers) != (RATE, WIDTH, 12):
            raise ValueError("Unexpected MERT-v1 geometry.")
        torch.set_num_threads(4)
        torch.manual_seed(2026)
        self.model = module.MERTModel(config)
        state = torch.load(DEST / "pytorch_model.bin", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.requires_grad_(False).eval().to(device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(DEST, local_files_only=True)
        if self.processor.sampling_rate != RATE or not self.processor.do_normalize:
            raise ValueError("Unexpected MERT-v1 preprocessing settings.")

    def extract_layers(self, path):
        chunks, vectors = load_v1_chunks(path), []
        with self.torch.inference_mode():
            for chunk in chunks:
                inputs = self.processor(chunk, sampling_rate=RATE, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                states = self.model(**inputs, output_hidden_states=True).hidden_states
                if len(states) != LAYERS:
                    raise ValueError("Expected embedding output plus 12 transformer layers.")
                vector = self.torch.stack(states).mean(dim=2).squeeze(1)
                values = vector.cpu().numpy().astype(np.float64)
                if values.shape != (LAYERS, WIDTH) or not np.isfinite(values).all():
                    raise ValueError("Invalid MERT-v1 layer output.")
                vectors.append(values)
        return summarize_v1_chunks(np.array(vectors))
