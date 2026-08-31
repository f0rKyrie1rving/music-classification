"""Layer-wise features for the frozen expanded MERT experiment."""

import numpy as np

from mert_features import CHUNKS, WIDTH, MertEncoder, load_chunks


LAYERS = 13


def summarize_layer_chunks(values):
    """Average six time-pooled chunk matrices without changing layer order."""
    values = np.asarray(values)
    if values.shape != (CHUNKS, LAYERS, WIDTH) or not np.isfinite(values).all():
        raise ValueError(f"Expected finite {(CHUNKS, LAYERS, WIDTH)} layer chunks.")
    return values.mean(axis=0, dtype=np.float64)


class LayerMertEncoder(MertEncoder):
    def extract_layers(self, path):
        chunks, vectors = load_chunks(path), []
        with self.torch.inference_mode():
            for chunk in chunks:
                inputs = self.processor(chunk, sampling_rate=16000, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                states = self.model(**inputs, output_hidden_states=True).hidden_states
                if len(states) != LAYERS:
                    raise ValueError("Expected embedding output plus 12 transformer layers.")
                # [13 layers, 1 example, frames, 768] -> [13, 768]
                vector = self.torch.stack(states).mean(dim=2).squeeze(1)
                values = vector.cpu().numpy().astype(np.float64)
                if values.shape != (LAYERS, WIDTH) or not np.isfinite(values).all():
                    raise ValueError("Invalid layer-wise encoder output.")
                vectors.append(values)
        return summarize_layer_chunks(np.array(vectors))
