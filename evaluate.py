import json

from embed_store import embed_texts
from generate_eval import normalize
from search import load_index, rank

EVAL_FILE = "eval_questions.json"
K_VALUES = [1, 3, 5]
MAX_K = max(K_VALUES)


def load_questions(path):
    """Load the list of {question, evidence} items."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_hit_rank(results, evidence_list):
    """Return the 1-based position of the first result containing any evidence phrase, or None."""
    for pos, r in enumerate(results, start=1):
        text = normalize(r["text"])
        for e in evidence_list:
            if normalize(e) in text:
                return pos
    return None


def hit_at_k(hit_ranks, k):
    """Fraction of questions whose hit rank is k or better."""
    hits = 0
    for hit in hit_ranks:
        if hit != None and hit <= k:
            hits += 1
    return hits / len(hit_ranks)


if __name__ == "__main__":
    chunks, vectors = load_index()
    items = load_questions(EVAL_FILE)

    questions = [item["question"] for item in items]
    q_vectors = embed_texts(questions, "RETRIEVAL_QUERY")

    hit_ranks = []
    misses = []
    for item, q in zip(items, q_vectors):
        results = rank(q, chunks, vectors, k=MAX_K)
        hr = find_hit_rank(results, item["evidence"])
        hit_ranks.append(hr)
        if hr is None:
            misses.append((item, results[0]["id"]))

    print(f"Evaluated {len(items)} questions from {EVAL_FILE}\n")
    for k in K_VALUES:
        print(f"hit@{k}: {hit_at_k(hit_ranks, k):.1%}")

    print(f"\nMisses (not in top {MAX_K}): {len(misses)}")
    for item, top_id in misses:
        print(f"  Q: {item['question']}")
        print(f"     evidence: {item['evidence']}")
        print(f"     top result was: {top_id}")
