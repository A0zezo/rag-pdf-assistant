import os


class Settings:
    """Runtime configuration, overridable via environment variables."""

    data_dir: str = os.getenv("RAG_DATA_DIR", "data")
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
    dense_top_k: int = int(os.getenv("RAG_DENSE_TOP_K", "8"))
    sparse_top_k: int = int(os.getenv("RAG_SPARSE_TOP_K", "8"))
    final_top_k: int = int(os.getenv("RAG_FINAL_TOP_K", "4"))
    rrf_k: int = int(os.getenv("RAG_RRF_K", "60"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


settings = Settings()
