# RAG PDF Assistant

Ask questions over your PDFs and get answers with page-level citations.
Hybrid retrieval (dense FAISS vectors + BM25 keyword search, fused with Reciprocal Rank Fusion), a FastAPI backend, and a small chat web UI. Runs fully local on CPU - no API key required.

![Chat UI answering a question with citations](https://github.com/user-attachments/assets/b843a6fb-0a67-4b62-83eb-a07105556d92)

## Features

- **PDF ingestion with page tracking** - text is extracted per page (`pypdf`) and every chunk keeps its source file and page number, so answers cite where they came from.
- **Hybrid retrieval** - dense semantic search (FAISS `IndexFlatIP` over L2-normalized `bge-small-en-v1.5` embeddings via fastembed/ONNX) combined with BM25+ keyword search, fused with Reciprocal Rank Fusion (RRF).
- **Answers with citations** - every answer returns its source passages, file names and page numbers.
- **Two answer modes** - extractive by default (returns the top retrieved passage, clearly labeled), or generated answers with inline citations when `OPENAI_API_KEY` is set.
- **Persistent index** - FAISS index + JSONL metadata are written to disk and survive restarts; new PDFs are added incrementally.
- **FastAPI + web UI** - REST API with auto-generated Swagger docs, plus a minimal chat UI with PDF upload. Shareable question links via `?q=...`.
- **Docker + tests + CI** - `Dockerfile`, `docker-compose.yml`, a `pytest` suite, and a GitHub Actions workflow that runs the tests on every push.

## Architecture

```
            ingest                          query
              │                               │
        ┌─────▼─────┐                   ┌─────▼─────┐
        │ PDF loader │                  │  Embedder │
        │  (pypdf)   │                  └─────┬─────┘
        └─────┬─────┘                         │
        ┌─────▼─────┐               ┌─────────▼─────────┐
        │  Chunker  │               │  Dense: FAISS     │
        │ (overlap, │               │  Sparse: BM25+    │
        │  metadata)│               └─────────┬─────────┘
        └─────┬─────┘                         │
        ┌─────▼──────────┐          ┌─────────▼─────────┐
        │ Embed → FAISS  │          │ RRF fusion → top-k │
        │ + chunks.jsonl │          └─────────┬─────────┘
        └────────────────┘          ┌─────────▼─────────┐
                                    │ Answer + citations │
                                    └───────────────────┘
```

## Quickstart

```bash
git clone https://github.com/A0zezo/rag-pdf-assistant.git
cd rag-pdf-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the API + web UI
uvicorn app.main:app --reload
# open http://localhost:8000        (chat UI, upload a PDF there)
# open http://localhost:8000/docs   (Swagger API docs)
```

Or try it from the CLI without the server:

```bash
python -m scripts.demo path/to/paper.pdf "What is the main contribution of this paper?"
```

With Docker:

```bash
docker compose up --build
```

## API

`POST /api/ingest` - upload a PDF (multipart form field `file`). Returns pages and chunk counts.

`POST /api/query` - body `{"question": "...", "top_k": 4}`. Example response (truncated):

```json
{
  "answer": "Most relevant passage (no OPENAI_API_KEY set, ...) - from attention.pdf, page 7: ...",
  "mode": "extractive",
  "citations": [
    {"source": "attention.pdf", "page": 7, "score": 0.0328},
    {"source": "attention.pdf", "page": 1, "score": 0.0301}
  ]
}
```

![Swagger API docs](https://github.com/user-attachments/assets/6b9d34ee-5e84-4382-af36-b15437cb4e4e)

## Demo (real output)

Ingesting `Attention Is All You Need` (arXiv:1706.03762), 15 pages -> 62 chunks:

```
Q: Which datasets are used for evaluation?
A (extractive): ... from attention.pdf, page 7:
   "... We trained on the standard WMT 2014 English-German dataset
    consisting of about 4.5 million sentence pairs ..."
Citations: [attention.pdf p.7, p.1, p.2, p.11]

Q: What optimizer did they use and what was the learning rate schedule?
A (extractive): ... from attention.pdf, page 7:
   "... 5.2 Hardware and Schedule - We trained our models on one machine
    with 8 NVIDIA P100 GPUs ..."
Citations: [attention.pdf p.7, p.7, p.1, p.15]
```

## Configuration (environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_DATA_DIR` | `data` | Where the FAISS index and metadata live |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `800` / `150` | Chunking, in characters |
| `RAG_DENSE_TOP_K` / `RAG_SPARSE_TOP_K` / `RAG_FINAL_TOP_K` | `8` / `8` / `4` | Retrieval depths |
| `RAG_RRF_K` | `60` | RRF smoothing constant |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | unset / `gpt-4o-mini` | Optional generated answers |

## Project structure

```
app/
  main.py        FastAPI app: /api/ingest, /api/query, /api/health, web UI
  pdf_loader.py  Page-by-page text extraction (pypdf)
  chunker.py     Recursive character chunking with overlap + metadata
  embeddings.py  fastembed (ONNX) embedder, L2-normalized vectors
  store.py       FAISS index + JSONL metadata persistence
  bm25.py        BM25+ sparse index (rank_bm25)
  retriever.py   Hybrid dense+sparse retrieval, RRF fusion
  generator.py   Answer building: OpenAI when configured, extractive otherwise
  static/        Chat web UI
scripts/demo.py  CLI demo: ingest a PDF and ask questions
tests/           pytest suite (chunker, store, BM25, RRF)
```

## Honest limitations

- Without `OPENAI_API_KEY` the "answer" is the top retrieved passage, verbatim and labeled as such - not a generated answer.
- Scanned PDFs (images, no text layer) are rejected; they need OCR first.
- The BM25 index is rebuilt in memory at startup, which suits personal-scale corpora (thousands of chunks), not millions.
- The embedding model is English-focused (`bge-small-en-v1.5`).

## Tech stack

Python 3.10+, FastAPI, FAISS, fastembed (ONNX Runtime), rank-bm25, pypdf, Docker, pytest, GitHub Actions.
