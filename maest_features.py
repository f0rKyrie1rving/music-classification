"""Pinned Discogs-MAEST layer-7 embedding features with no network use."""

import json
from pathlib import Path

import numpy as np

from prepare_maest import (DEST, METADATA_NAME, MODEL_NAME, EXPECTED_SIZES,
                           sha256, validate_metadata, verify_file)


RATE, SECONDS, SAMPLES = 16_000, 30, 480_000
TOKEN_WIDTH, FEATURE_WIDTH = 768, 2_304
OUTPUT = "PartitionedCall/Identity_7"


def validate_audio(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or len(audio) < SAMPLES:
        raise ValueError("MAEST input must contain at least 30 seconds of mono audio.")
    audio = audio[:SAMPLES]
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples.")
    if np.sqrt(np.mean(audio.astype(np.float64) ** 2)) < 1e-5:
        raise ValueError("Audio is silent or below the RMS threshold (1e-5).")
    return audio


def load_maest_audio(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    import essentia.standard as es
    try:
        audio = es.MonoLoader(filename=str(path), sampleRate=RATE, resampleQuality=4)()
    except RuntimeError as error:
        raise ValueError(f"Invalid audio: {error}") from error
    return validate_audio(audio)


def summarize_maest_embeddings(values):
    values = np.asarray(values)
    while values.ndim > 2 and values.shape[0] == 1:
        values = values[0]
    if (values.ndim != 2 or values.shape[0] < 3 or values.shape[1] != TOKEN_WIDTH
            or not np.isfinite(values).all()):
        raise ValueError("Expected finite MAEST token embeddings with width 768.")
    vector = np.concatenate((values[0], values[1], values[2:].mean(axis=0, dtype=np.float64)))
    if vector.shape != (FEATURE_WIDTH,) or not np.isfinite(vector).all():
        raise ValueError("Invalid MAEST CLS/DIST/signal-mean feature vector.")
    return vector


class MaestEncoder:
    def __init__(self):
        import essentia.standard as es

        receipt_path = DEST / "SOURCES.json"
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("model") != "discogs-maest-30s-pw-519l" or receipt.get("version") != "2":
            raise ValueError("MAEST source receipt differs from the frozen candidate.")
        for name in (METADATA_NAME, MODEL_NAME):
            digest = verify_file(DEST / name, EXPECTED_SIZES[name])
            if digest != receipt["files"][name]["observed_sha256"]:
                raise ValueError(f"MAEST source mismatch: {name}")
        validate_metadata(DEST / METADATA_NAME)
        self.model_sha256 = receipt["files"][MODEL_NAME]["observed_sha256"]
        self.model = es.TensorflowPredictMAEST(
            graphFilename=str(DEST / MODEL_NAME), output=OUTPUT)

    def extract(self, path):
        audio = load_maest_audio(path)
        return summarize_maest_embeddings(self.model(audio))
