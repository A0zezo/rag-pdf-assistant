"""End-to-end demo: ingest a PDF and ask questions against it, without the server.

Usage: python -m scripts.demo path/to/file.pdf ["question 1" "question 2" ...]
"""

import sys

from app.chunker import chunk_pages
from app.config import settings
from app.embeddings import Embedder
from app.generator import citations_of, generate_answer
from app.pdf_loader import load_pdf
from app.retriever import HybridRetriever
from app.store import VectorStore

DEFAULT_QUESTIONS = [
    "What is the main contribution of this paper?",
    "Which datasets are used for evaluation?",
]


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    pdf_path = sys.argv[1]
    questions = sys.argv[2:] or DEFAULT_QUESTIONS

    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages, source=pdf_path.split("/")[-1], size=settings.chunk_size, overlap=settings.chunk_overlap)
    print(f"Ingested {len(pages)} pages -> {len(chunks)} chunks")

    embedder = Embedder.get()
    store = VectorStore()
    store.add(embedder.embed([c["text"] for c in chunks]), chunks)
    retriever = HybridRetriever(store, embedder)

    for question in questions:
        print(f"\nQ: {question}")
        contexts = retriever.retrieve(question)
        answer, mode = generate_answer(question, contexts)
        print(f"A ({mode}): {answer}")
        print("Citations:", citations_of(contexts))


if __name__ == "__main__":
    main()
