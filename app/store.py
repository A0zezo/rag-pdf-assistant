import json
import os
import threading

import faiss
import numpy as np

from .config import settings


class VectorStore:
    """FAISS inner-product index over normalized vectors, with JSONL metadata sidecar."""

    def __init__(self, data_dir: str | None = None):
        self.data_dir = data_dir or settings.data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.index_path = os.path.join(self.data_dir, "index.faiss")
        self.meta_path = os.path.join(self.data_dir, "chunks.jsonl")
        self.lock = threading.Lock()
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[dict] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, encoding="utf-8") as f:
                self.chunks = [json.loads(line) for line in f if line.strip()]

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def sources(self) -> list[str]:
        return sorted({c["source"] for c in self.chunks})

    def add(self, vectors: np.ndarray, metadatas: list[dict]) -> list[int]:
        with self.lock:
            if self.index is None:
                self.index = faiss.IndexFlatIP(vectors.shape[1])
            start = len(self.chunks)
            self.index.add(vectors)
            for offset, meta in enumerate(metadatas):
                record = dict(meta)
                record["id"] = start + offset
                self.chunks.append(record)
            self._save()
            return [start + offset for offset in range(len(metadatas))]

    def _save(self) -> None:
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self.index is None or not self.chunks:
            return []
        scores, ids = self.index.search(query_vector, min(top_k, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
