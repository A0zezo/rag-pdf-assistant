import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .chunker import chunk_pages
from .config import settings
from .embeddings import Embedder
from .generator import citations_of, generate_answer
from .pdf_loader import load_pdf
from .retriever import HybridRetriever
from .store import VectorStore

app = FastAPI(title="RAG PDF Assistant", version="1.0.0")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
_state: dict = {}


def pipeline() -> dict:
    if "retriever" not in _state:
        store = VectorStore()
        embedder = Embedder.get()
        _state["store"] = store
        _state["embedder"] = embedder
        _state["retriever"] = HybridRetriever(store, embedder)
    return _state


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


@app.get("/api/health")
def health():
    store = VectorStore()
    return {"status": "ok", "chunks_indexed": store.size, "sources": store.sources}


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    state = pipeline()
    uploads_dir = os.path.join(settings.data_dir, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    dest = os.path.join(uploads_dir, os.path.basename(file.filename))
    with open(dest, "wb") as f:
        f.write(await file.read())
    try:
        pages = load_pdf(dest)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    chunks = chunk_pages(pages, source=file.filename, size=settings.chunk_size, overlap=settings.chunk_overlap)
    vectors = state["embedder"].embed([c["text"] for c in chunks])
    state["store"].add(vectors, chunks)
    state["retriever"].refresh()
    return {
        "file": file.filename,
        "pages": len(pages),
        "chunks_indexed": len(chunks),
        "total_chunks": state["store"].size,
    }


@app.post("/api/query")
def query(request: QueryRequest):
    state = pipeline()
    if state["store"].size == 0:
        raise HTTPException(status_code=400, detail="Index is empty. Ingest a PDF first.")
    contexts = state["retriever"].retrieve(request.question, final_top_k=request.top_k)
    answer, mode = generate_answer(request.question, contexts)
    return {"answer": answer, "mode": mode, "citations": citations_of(contexts), "contexts": contexts}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
