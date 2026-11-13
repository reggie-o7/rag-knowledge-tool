import json
import random
from pathlib import Path

from google.genai import types

from answer import generate_with_retry
from search import load_index

NUM_CANDIDATES = 50
MIN_WORDS = 100  # skip short leftover chunks at the end of files
SEED = 42  # same random picks every run
OUTPUT_FILE = "eval_candidates.json"
EVAL_MODEL = "gemini-3.5-flash-lite"

QUESTION_PROMPT = """Below is a passage from a document about Lionel Messi.

Write ONE question that a curious person might ask, which this passage answers.

Rules:
- The question must make sense on its own, without seeing the passage. Never say "in this passage" or "according to the text".
- Ask about ONE specific fact with a clear answer: a name, number, date, team, or event.
- Include enough detail that the question is unambiguous (for example, name the tournament or year).
- Use your own wording. Do not copy phrases from the passage.
- Also give an evidence phrase: 5 to 12 words copied EXACTLY, character for character, from the passage, containing the answer.

Respond with JSON only, in this format:
{{"question": "...", "evidence": "..."}}

Passage:
{passage}"""


def normalize(text):
    """Collapse whitespace and lowercase, so small spacing/case differences don't matter."""
    return " ".join(text.split()).lower()


def pick_chunks(chunks, n, min_words, seed):
    """Randomly choose n chunks that have at least min_words words."""
    eligible = [c for c in chunks if len(c["text"].split()) >= min_words]
    rng = random.Random(seed)
    return rng.sample(eligible, n)


def generate_candidate(chunk):
    """Ask Gemini for a question + evidence phrase about one chunk. Return a dict."""
    prompt = QUESTION_PROMPT.format(passage=chunk["text"])
    config = types.GenerateContentConfig(response_mime_type="application/json")
    response = generate_with_retry(prompt, config, model=EVAL_MODEL)
    if response.text is None:
        raise RuntimeError(f"No text returned for {chunk['id']}")
    return json.loads(response.text)


def evidence_is_valid(evidence, chunk_text):
    """Return True if the evidence phrase really appears in the chunk text."""
    return normalize(evidence) in normalize(chunk_text)


def load_candidates(path):
    """Load previously saved candidates, or return an empty list if none exist yet."""
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_candidates(candidates, path):
    """Write all candidates to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    chunks, _ = load_index()
    picked = pick_chunks(chunks, NUM_CANDIDATES, MIN_WORDS, SEED)

    candidates = load_candidates(OUTPUT_FILE)
    done = {c["source_chunk"] for c in candidates}
    rejected = 0

    for i, chunk in enumerate(picked, start=1):
        if chunk["id"] in done:
            continue
        cand = generate_candidate(chunk)
        if evidence_is_valid(cand["evidence"], chunk["text"]):
            candidates.append(
                {
                    "question": cand["question"],
                    "evidence": [cand["evidence"]],
                    "source_chunk": chunk["id"],
                }
            )
            save_candidates(candidates, OUTPUT_FILE)
            print(f"[{i}/{len(picked)}] OK   {cand['question']}")
        else:
            rejected += 1
            print(f"[{i}/{len(picked)}] BAD  evidence not found in {chunk['id']}")

    print(
        f"\nTotal saved: {len(candidates)}, rejected this run: {rejected}. File: {OUTPUT_FILE}"
    )
