from .bm25 import SparseIndex
from .config import settings
from .embeddings import Embedder
from .store import VectorStore


def rrf_fuse(result_lists: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: combine ranked lists without comparing raw scores."""
    scores: dict[int, float] = {}
    for results in result_lists:
        for rank, (chunk_id, _) in enumerate(results):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """Dense (FAISS) + sparse (BM25) retrieval fused with RRF."""

    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder
        self.sparse = SparseIndex()
        self.refresh()

    def refresh(self) -> None:
        self.sparse.build(self.store.chunks)

    def retrieve(
        self,
        question: str,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        final_top_k: int | None = None,
    ) -> list[dict]:
        dense_top_k = dense_top_k or settings.dense_top_k
        sparse_top_k = sparse_top_k or settings.sparse_top_k
        final_top_k = final_top_k or settings.final_top_k
        query_vector = self.embedder.embed([question])
        dense = self.store.search(query_vector, dense_top_k)
        sparse = self.sparse.search(question, sparse_top_k)
        fused = rrf_fuse([dense, sparse], k=settings.rrf_k)
        results = []
        for chunk_id, score in fused[:final_top_k]:
            chunk = self.store.chunks[chunk_id]
            results.append(
                {
                    "id": chunk_id,
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "score": round(score, 6),
                }
            )
        return results
