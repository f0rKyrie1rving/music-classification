"""Frozen MERT-v0-public features, using checked local files and no network calls."""

import contextlib
import hashlib
import importlib
import json
import os
import sys
import types
from pathlib import Path

import librosa
import numpy as np

from prepare_mert import DEST, FILES, REVISION, verify

ROOT = Path(__file__).resolve().parent
RATE, CHUNK_SECONDS, CHUNKS, WIDTH = 16000, 5, 6, 768
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache/numba"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def load_chunks(path):
    """This experimental encoder requires >=30 s; analyze only the first 30 s."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        audio, _ = librosa.load(path, sr=RATE, mono=True, offset=0, duration=30,
                                dtype=np.float32, res_type="soxr_hq")
    except librosa.util.exceptions.ParameterError as error:
        raise ValueError(f"Invalid audio: {error}") from error
    if len(audio) != RATE * CHUNK_SECONDS * CHUNKS:
        raise ValueError("Experimental MERT input must contain at least 30 seconds.")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples.")
    if np.sqrt(np.mean(audio.astype(np.float64) ** 2)) < 1e-5:
        raise ValueError("Audio is silent or below the RMS threshold (1e-5).")
    return audio.reshape(CHUNKS, RATE * CHUNK_SECONDS)


class MertEncoder:
    def __init__(self, device="cpu"):
        # torch is optional for the original MFCC environment and unit tests.
        import torch
        from transformers import Wav2Vec2FeatureExtractor

        self.torch, self.device = torch, device
        if device not in ("cpu", "mps") or (device == "mps" and not torch.backends.mps.is_available()):
            raise ValueError(f"Unavailable encoder device: {device}")
        sources = json.loads((DEST / "SOURCES.json").read_text())
        if sources["revision"] != REVISION or set(sources["files_sha256"]) != set(FILES):
            raise ValueError("Encoder source receipt differs from the pinned version.")
        for name, expected in FILES.items():
            if verify(DEST / name, expected) != sources["files_sha256"][name]:
                raise ValueError(f"Encoder source mismatch: {name}")
        # Import reviewed local code explicitly; never use moving remote code.
        package = types.ModuleType("portfolio_mert")
        package.__path__ = [str(DEST)]
        sys.modules[package.__name__] = package
        with contextlib.redirect_stdout(sys.stderr):
            module = importlib.import_module("portfolio_mert.modeling_MERT")
        config = module.MERTConfig(**json.loads((DEST / "config.json").read_text()))
        if (config.sample_rate, config.hidden_size, config.num_hidden_layers) != (RATE, WIDTH, 12):
            raise ValueError("Unexpected encoder geometry.")
        torch.set_num_threads(4)
        torch.manual_seed(2026)
        self.model = module.MERTModel(config)
        state = torch.load(DEST / "pytorch_model.bin", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.requires_grad_(False).eval().to(device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(DEST, local_files_only=True)
        if self.processor.sampling_rate != RATE or self.processor.do_normalize:
            raise ValueError("Unexpected preprocessor settings.")

    def extract(self, path):
        chunks, vectors = load_chunks(path), []
        with self.torch.inference_mode():
            for chunk in chunks:
                inputs = self.processor(chunk, sampling_rate=RATE, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                states = self.model(**inputs, output_hidden_states=True).hidden_states
                if len(states) != 13:
                    raise ValueError("Expected embedding layer plus 12 transformer layers.")
                # [12 layers, 1 example, frames, 768]; exclude embedding layer 0.
                vector = self.torch.stack(states[1:]).mean(dim=(0, 2)).squeeze(0)
                vectors.append(vector.cpu().numpy().astype(np.float64))
        values = np.array(vectors)
        if values.shape != (CHUNKS, WIDTH) or not np.isfinite(values).all():
            raise ValueError("Invalid encoder output.")
        return values.mean(axis=0)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
