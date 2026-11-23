import httpx

from .config import settings

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context below.
Cite sources inline as (source, page). If the context does not contain the answer, say so.

Context:
{context}

Question: {question}
Answer:"""


def _format_context(contexts: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(contexts, 1):
        blocks.append(f"[{i}] ({c['source']}, page {c['page']})\n{c['text']}")
    return "\n\n".join(blocks)


def citations_of(contexts: list[dict]) -> list[dict]:
    return [{"source": c["source"], "page": c["page"], "score": c["score"]} for c in contexts]


def _openai_answer(question: str, contexts: list[dict]) -> str:
    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.openai_api_key}"},
        json={
            "model": settings.openai_model,
            "messages": [
                {"role": "user", "content": PROMPT_TEMPLATE.format(context=_format_context(contexts), question=question)}
            ],
            "temperature": 0,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def _extractive_answer(contexts: list[dict]) -> str:
    """Fallback with no LLM key: return the best passage verbatim, clearly labeled."""
    top = contexts[0]
    text = top["text"].strip()
    if len(text) > 600:
        text = text[:600].rsplit(" ", 1)[0] + " ..."
    return (
        "Most relevant passage (no OPENAI_API_KEY set, so this is the top retrieved "
        f"context, not a generated answer) - from {top['source']}, page {top['page']}:\n\n{text}"
    )


def generate_answer(question: str, contexts: list[dict]) -> tuple[str, str]:
    """Return (answer, mode). Mode is 'openai' or 'extractive'."""
    if not contexts:
        return "No relevant passages found in the indexed documents.", "extractive"
    if settings.openai_api_key:
        return _openai_answer(question, contexts), "openai"
    return _extractive_answer(contexts), "extractive"
