import json
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from chunker import CHUNK_SIZE, build_chunks

load_dotenv()
client = genai.Client()

MODEL = "gemini-embedding-001"
BATCH_SIZE = 50
INDEX_DIR = Path(f"index_{CHUNK_SIZE}")
MAX_RETRIES = 5
BASE_DELAY = 15
RETRYABLE_CODES = {429, 500, 503}

def embed_batch(batch, task_type):
    """Embed one batch, retrying with exponential backoff on temporary API errors."""
    delay = BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.models.embed_content(
                model=MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
        except errors.APIError as e:
            if e.code not in RETRYABLE_CODES or attempt == MAX_RETRIES:
                raise
            print(f"API error {e.code} (attempt {attempt}/{MAX_RETRIES}). Waiting {delay}s...")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("embed_batch: retries exhausted")


def embed_texts(texts, task_type):
    """Embed a list of strings in batches. Return a NumPy array, one row per text."""
    all_vectors = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]

        result = embed_batch(batch, task_type)

        if result.embeddings is None:
            raise RuntimeError(f"No embeddings returned for batch starting at index {i}")

        for e in result.embeddings:
            all_vectors.append(e.values)

    assert len(all_vectors) == len(texts), f"Got {len(all_vectors)} vectors for {len(texts)} texts"
    return np.array(all_vectors)


def save_index(chunks, vectors):
    """Save chunks to index/chunks.json and vectors to index/embeddings.npy."""
    INDEX_DIR.mkdir(exist_ok=True)
    with open(INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    np.save(INDEX_DIR / "embeddings.npy", vectors)


if __name__ == "__main__":
    chunks = build_chunks()
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts, "RETRIEVAL_DOCUMENT")
    print("Embeddings shape:", vectors.shape)
    save_index(chunks, vectors)
    print("Saved index to", INDEX_DIR)