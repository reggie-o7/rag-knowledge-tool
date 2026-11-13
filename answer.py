import time

from google.genai import errors

from embed_store import BASE_DELAY, MAX_RETRIES, RETRYABLE_CODES, client
from search import load_index, search

GEN_MODEL = "gemini-3.1-flash-lite"

INSTRUCTIONS = """You answer questions about Lionel Messi using ONLY the sources below.

Rules:
- Use only information from the sources. Do not use outside knowledge.
- After each fact, cite its source id in square brackets, like [messi_main.txt#16].
- If the sources do not contain the answer, reply exactly: I couldn't find that in the sources.
- Keep the answer to 1-3 sentences."""


def build_prompt(question, results):
    """Combine instructions, retrieved chunks, and the question into one prompt string."""
    blocks = []
    for r in results:
        blocks.append(f"[{r['id']}]\n{r['text']}")
    sources = "\n\n".join(blocks)

    return f"{INSTRUCTIONS}\n\nSources:\n{sources}\n\nQuestion: {question}\nAnswer:"


def generate_with_retry(prompt, config=None, model=GEN_MODEL):
    """Generate a response, retrying on temporary API errors."""
    delay = BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        except errors.APIError as e:
            if e.code not in RETRYABLE_CODES or attempt == MAX_RETRIES:
                raise
            print(f"API error {e.code} (attempt {attempt}/{MAX_RETRIES}). Waiting {delay}s...")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("generate_with_retry: retries exhausted")


def answer(question, chunks, vectors):
    """Retrieve the top chunks, have Gemini answer from them, and return the answer with its sources."""
    results = search(question, chunks, vectors)
    prompt = build_prompt(question, results)
    response = generate_with_retry(prompt)

    if response.text is None:
        raise RuntimeError(f"Gemini returned no text for question: {question!r}")

    sources = []
    for r in results:
        sources.append({"id": r["id"], "score": float(r["score"])})

    return {"answer": response.text, "sources": sources}


if __name__ == "__main__":
    chunks, vectors = load_index()

    questions = [
        "Who did Argentina beat in the 2022 World Cup final?",
        "How many Ballon d'Or awards has Messi won?",
        "When did Messi join Inter Miami?",
        "What is Messi's favorite food?",
    ]

    for q in questions:
        print("Q:", q)
        print("A:", answer(q, chunks, vectors)["answer"])
        print()