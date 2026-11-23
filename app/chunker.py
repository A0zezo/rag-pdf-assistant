from .pdf_loader import Page

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    step = max(size - overlap, 1)
    return [text[i : i + size].strip() for i in range(0, len(text), step) if text[i : i + size].strip()]


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    """Recursive character splitter: try paragraph, line, sentence, then word boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    sep = next((s for s in SEPARATORS if s in text), None)
    if sep is None:
        return _hard_split(text, size, overlap)
    chunks: list[str] = []
    current = ""
    for piece in text.split(sep):
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) > size:
            if current:
                chunks.append(current)
                current = ""
            if sep == SEPARATORS[-1]:
                chunks.extend(_hard_split(piece, size, overlap))
            else:
                chunks.extend(_split_text(piece, size, overlap))
            continue
        candidate = f"{current}{sep}{piece}" if current else piece
        if len(candidate) <= size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    if overlap and len(chunks) > 1:
        merged = [chunks[0]]
        for prev, cur in zip(chunks, chunks[1:]):
            merged.append(f"{prev[-overlap:]} {cur}".strip())
        chunks = merged
    return chunks


def chunk_pages(pages: list[Page], source: str, size: int, overlap: int) -> list[dict]:
    """Split pages into overlapping chunks, each carrying source + page metadata."""
    chunks = []
    for page in pages:
        for text in _split_text(page.text, size, overlap):
            chunks.append({"text": text, "source": source, "page": page.number})
    return chunks
