import re

from rank_bm25 import BM25Plus


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


class SparseIndex:
    """BM25+ keyword index over the same chunks the dense index holds."""

    def __init__(self):
        self.bm25: BM25Plus | None = None
        self.chunk_ids: list[int] = []

    def build(self, chunks: list[dict]) -> None:
        self.chunk_ids = [c["id"] for c in chunks]
        corpus = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Plus(corpus) if corpus else None

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.chunk_ids[i], float(s)) for i, s in ranked if s > 0]
