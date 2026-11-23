from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class Page:
    number: int  # 1-based page number
    text: str


def load_pdf(path: str) -> list[Page]:
    """Extract text page by page so every chunk can cite its page."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Page(number=i + 1, text=text))
    if not pages:
        raise ValueError(
            "No extractable text found in this PDF. Scanned documents need OCR first."
        )
    return pages
