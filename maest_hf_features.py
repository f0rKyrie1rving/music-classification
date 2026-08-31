"""Pinned official MTG-UPF MAEST features with reviewed local preprocessing."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from prepare_maest_hf import DEST, FILES, MODEL_ID, REVISION
from prepare_mert import verify


ROOT = Path(__file__).resolve().parent
RATE, SECONDS, SAMPLES = 16_000, 30, 480_000
TOKEN_WIDTH, FEATURE_WIDTH, HIDDEN_STATE_INDEX = 768, 2_304, 7
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def load_audio(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        with sf.SoundFile(path) as source:
            source_rate = source.samplerate
            samples = source.read(frames=SECONDS * source_rate, dtype="float32", always_2d=True)
    except RuntimeError as error:
        raise ValueError(f"Invalid audio: {error}") from error
    if len(samples) != SECONDS * source_rate:
        raise ValueError("MAEST input must contain at least 30 seconds.")
    audio = samples.mean(axis=1)
    if source_rate != RATE:
        audio = soxr.resample(audio, source_rate, RATE, quality="HQ")
    if len(audio) != SAMPLES:
        raise ValueError("MAEST resampling did not produce exactly 30 seconds.")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples.")
    if np.sqrt(np.mean(audio.astype(np.float64) ** 2)) < 1e-5:
        raise ValueError("Audio is silent or below the RMS threshold (1e-5).")
    return np.asarray(audio, dtype=np.float32)


def summarize_hidden_state(values):
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if (values.ndim != 2 or values.shape[0] < 3 or values.shape[1] != TOKEN_WIDTH
            or not np.isfinite(values).all()):
        raise ValueError("Expected finite MAEST hidden tokens with width 768.")
    vector = np.concatenate((values[0], values[1], values[2:].mean(axis=0, dtype=np.float64)))
    if vector.shape != (FEATURE_WIDTH,) or not np.isfinite(vector).all():
        raise ValueError("Invalid MAEST CLS/DIST/signal-mean feature vector.")
    return vector


class HfMaestEncoder:
    def __init__(self, device="cpu"):
        import torch
        from transformers import ASTForAudioClassification

        self.torch, self.device = torch, device
        if device not in ("cpu", "mps") or (device == "mps" and not torch.backends.mps.is_available()):
            raise ValueError(f"Unavailable encoder device: {device}")
        receipt = json.loads((DEST / "SOURCES.json").read_text())
        if (receipt.get("model_id") != MODEL_ID or receipt.get("revision") != REVISION
                or set(receipt.get("files_sha256", {})) != set(FILES)):
            raise ValueError("MAEST source receipt differs from the pinned repository.")
        for name, expected in FILES.items():
            if verify(DEST / name, expected) != receipt["files_sha256"][name]:
                raise ValueError(f"MAEST source mismatch: {name}")

        spec = importlib.util.spec_from_file_location(
            "portfolio_maest_feature_extractor", DEST / "feature_extraction_maest.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        settings = json.loads((DEST / "preprocessor_config.json").read_text())
        settings.pop("auto_map", None); settings.pop("feature_extractor_type", None)
        self.processor = module.MAESTFeatureExtractor(**settings)
        geometry = (self.processor.sampling_rate, self.processor.num_mel_bins,
                    self.processor.max_length, self.processor.n_fft,
                    self.processor.hop_length, self.processor.do_normalize)
        if geometry != (RATE, 96, 1876, 512, 256, True):
            raise ValueError("Unexpected MAEST feature-extractor settings.")

        torch.set_num_threads(4); torch.manual_seed(2026)
        self.model = ASTForAudioClassification.from_pretrained(
            DEST, local_files_only=True, use_safetensors=True)
        config = self.model.config
        if (config.hidden_size, config.num_hidden_layers, config.num_attention_heads,
                config.num_mel_bins, config.max_length, config.num_labels) != (768, 12, 12, 96, 1876, 519):
            raise ValueError("Unexpected MAEST model geometry.")
        self.model.requires_grad_(False).eval().to(device)

    def extract(self, path):
        audio = load_audio(path)
        inputs = self.processor(audio, sampling_rate=RATE, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            result = self.model(**inputs, output_hidden_states=True, return_dict=True)
        if len(result.hidden_states) != 13:
            raise ValueError("Expected embedding output plus 12 MAEST transformer layers.")
        hidden = result.hidden_states[HIDDEN_STATE_INDEX].cpu().numpy().astype(np.float64)
        return summarize_hidden_state(hidden)
