import numpy as np

from app.bm25 import SparseIndex
from app.chunker import chunk_pages
from app.pdf_loader import Page
from app.retriever import rrf_fuse
from app.store import VectorStore


def test_chunker_respects_size_and_keeps_pages():
    pages = [Page(number=1, text="word " * 500), Page(number=2, text="other " * 300)]
    chunks = chunk_pages(pages, source="toy.pdf", size=800, overlap=150)
    assert chunks, "expected chunks"
    assert all(c["source"] == "toy.pdf" for c in chunks)
    assert {c["page"] for c in chunks} == {1, 2}
    assert all(len(c["text"]) <= 800 + 150 + 1 for c in chunks)


def test_chunker_short_text_single_chunk():
    chunks = chunk_pages([Page(number=3, text="short text")], source="a.pdf", size=800, overlap=150)
    assert len(chunks) == 1
    assert chunks[0]["page"] == 3


def test_rrf_prefers_items_ranked_high_in_both_lists():
    dense = [(1, 0.9), (2, 0.8), (3, 0.7)]
    sparse = [(2, 5.0), (1, 4.0), (9, 3.0)]
    fused = rrf_fuse([dense, sparse], k=60)
    assert fused[0][0] in (1, 2)
    assert fused[-1][0] == 9


def test_vector_store_roundtrip(tmp_path):
    store = VectorStore(data_dir=str(tmp_path))
    vectors = np.eye(4, dtype="float32")
    ids = store.add(vectors, [{"text": f"chunk {i}", "source": "toy.pdf", "page": 1} for i in range(4)])
    assert ids == [0, 1, 2, 3]
    reloaded = VectorStore(data_dir=str(tmp_path))
    assert reloaded.size == 4
    hits = reloaded.search(np.array([[0, 1, 0, 0]], dtype="float32"), top_k=1)
    assert hits[0][0] == 1


def test_sparse_index_ranks_keyword_match_first():
    index = SparseIndex()
    index.build([
        {"id": 0, "text": "cats like milk and naps"},
        {"id": 1, "text": "transformer models use self attention layers"},
    ])
    hits = index.search("self attention", top_k=1)
    assert hits[0][0] == 1
