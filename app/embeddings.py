import numpy as np
from fastembed import TextEmbedding

from .config import settings


class Embedder:
    """Lazy singleton around fastembed (ONNX, runs on CPU, no GPU or API key needed)."""

    _instance: "Embedder | None" = None

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self.model = TextEmbedding(self.model_name)

    @classmethod
    def get(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(list(self.model.embed(texts)), dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms  # L2-normalized, so inner product == cosine similarity
