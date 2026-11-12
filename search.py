import json

import numpy as np

from embed_store import INDEX_DIR, embed_texts

TOP_K = 3


def load_index():
    """Load chunks and their embeddings from the index folder."""
    with open(INDEX_DIR / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    vectors = np.load(INDEX_DIR / "embeddings.npy")
    return chunks, vectors

def cosine_similarity(a, b):
    """Return the cosine similarity between two vectors (1 = same direction, 0 = unrelated)."""
    return (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def rank(q, chunks, vectors, k=TOP_K):
    """Return the k chunks most similar to an already-embedded question vector q."""
    scores = []
    for row in vectors:
        scores.append(cosine_similarity(q, row))
    
    scores = np.array(scores)

    top_pos = np.argsort(scores)[::-1][:k]

    results = []

    for idx in top_pos:
        results.append({
            "id": chunks[idx]["id"],
            "text": chunks[idx]["text"],
            "score": scores[idx],
        })

    return results

def search(question, chunks, vectors, k=TOP_K):
    """Embed a question, then return its top k chunks."""
    q = embed_texts([question], "RETRIEVAL_QUERY")[0]
    return rank(q, chunks, vectors, k)

if __name__ == "__main__":
    chunks, vectors = load_index()

    questions = [
        "Who did Argentina beat in the 2022 World Cup final?",
        "How many Ballon d'Or awards has Messi won?",
        "When did Messi join Inter Miami?",
    ]

    for q in questions:
        print("Q:", q)
        for r in search(q, chunks, vectors):
            print(f"   {r['score']:.3f}  {r['id']}")
            print(f"          {r['text'][:120]}...")
        print()