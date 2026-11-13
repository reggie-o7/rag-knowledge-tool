# RAG Knowledge Query Tool

A retrieval-augmented generation (RAG) system that answers questions about Lionel Messi from a local document corpus, with citations to the exact source chunks used. Built in Python with Gemini embeddings, NumPy vector search, and a FastAPI endpoint, and evaluated with a retrieval benchmark.

```
POST /ask  {"question": "When did Messi start playing for Miami?"}

→ {
    "answer": "Inter Miami CF announced the signing of Lionel Messi on 15 July 2023 [messi_main.txt#16]. He made his debut for the club on 21 July 2023 [inter_miami.txt#9].",
    "sources": [
      {"id": "inter_miami.txt#9",  "score": 0.767},
      {"id": "messi_main.txt#16",  "score": 0.761},
      {"id": "messi_main.txt#18",  "score": 0.746}
    ]
  }
```

## How it works

The system has two parts:

- **Indexing (done once):** documents → chunks → embeddings → saved to disk
- **Answering (every question):** question → embedding → find the 3 closest chunks → Gemini writes a cited answer from them

In more detail:

1. **Chunking**: documents are cleaned (Wikipedia citation markers removed) and split into 300-word chunks with a 50-word overlap, so facts near a boundary appear whole in at least one chunk.
2. **Embedding**: each chunk is embedded once with `gemini-embedding-001` (task type `RETRIEVAL_DOCUMENT`, 3072 dimensions) and saved to disk, so queries never re-embed the corpus.
3. **Retrieval**: the question is embedded (task type `RETRIEVAL_QUERY`), scored against every chunk with cosine similarity, and the top 3 chunks are kept.
4. **Generation**: the chunks are passed to Gemini with instructions to answer *only* from the sources, cite each fact by chunk id, and reply "I couldn't find that in the sources." when the answer isn't there.

## Project structure

| File | Purpose |
|---|---|
| `chunker.py` | Loads, cleans, and chunks the documents in `data/` |
| `embed_store.py` | Embeds all chunks in batches and saves the index; retry logic for API errors |
| `search.py` | Loads the index; `rank()` scores a query vector, `search()` embeds a question and ranks |
| `answer.py` | Builds the grounded prompt and generates a cited answer |
| `api.py` | FastAPI app exposing `POST /ask` and `GET /health` |
| `evaluate.py` | Measures retrieval quality (hit@1/3/5) on the evaluation set |
| `generate_eval.py` | Generates candidate evaluation questions with Gemini |
| `eval_questions.json` | The reviewed evaluation set (27 questions) |
| `check_setup.py` | Sanity check that the API key and embeddings work |
| `data/` | The six source documents |

## Setup

Requires Python 3.10+ and a [Gemini API key](https://aistudio.google.com/).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your-key-here
```

## Usage

```bash
python embed_store.py              # build the index (once, or after changing data/chunk settings)
python answer.py                   # run the sample questions in the terminal
uvicorn api:app --reload           # start the API at http://127.0.0.1:8000
python evaluate.py                 # measure retrieval quality
```

With the server running, open **http://127.0.0.1:8000/docs** for an interactive page to try the API.

## Evaluation

### Method

Retrieval is measured with **hit@k**: the share of questions where a chunk containing the answer appears in the top *k* results. hit@3 matters most, because the top 3 chunks are what the model sees.

- **Labels are evidence phrases, not chunk ids.** Each question lists short exact phrases from the source text that contain the answer; a retrieved chunk counts as a hit if it contains any of them. Because labels don't depend on chunk boundaries, the same question set can compare different chunking settings.
- **Facts that appear in multiple documents list evidence from each**, so retrieving a correct chunk from another file isn't scored as a miss.
- **The question set was LLM-generated and human-reviewed.** `generate_eval.py` asks Gemini for a question and an evidence phrase from randomly sampled chunks, and automatically discards any candidate whose evidence phrase doesn't appear verbatim in its chunk. Every surviving candidate was then reviewed by hand: trivia and ambiguous questions were removed, questions were reworded to avoid copying the source's phrasing, and evidence from other documents was added. 27 questions remain.

### Results

Chunk-size comparison, with everything else held fixed (same questions, embedding model, and scoring):

| Chunk size (overlap) | Chunks | hit@1 | hit@3 | hit@5 |
|---|---|---|---|---|
| **300 words (50)** | 143 | **59.3%** | **81.5%** | **96.3%** |
| 200 words (40) | 220 | 51.9% | 81.5% | 92.6% |

300-word chunks matched or beat 200-word chunks at every *k*, so 300 is the default. With 27 questions, each question is worth about 3.7 points, so these differences are small; the result supports "smaller chunks did not help" more than "300 is definitively better."

Both indexes are kept (`index_300/`, `index_200/`; the folder is named from `CHUNK_SIZE`), so either result can be reproduced by setting `CHUNK_SIZE` in `chunker.py` and running `python evaluate.py`.

### Failure analysis

The one miss at 300 words, *"How many assists did Messi provide in Ligue 1 during the 2022 calendar year?"*, has its answer on a single line inside a dense list of unrelated records (La Liga age records, French records, MLS awards). That chunk's embedding is an average of many topics, so it doesn't strongly match any one question. A prose chunk about Messi's PSG season ranked first instead: right topic, but it doesn't contain the figure.

The hypothesis was that smaller chunks would sharpen these list-heavy chunks. They didn't: the miss persisted at 200 words and a second list-based question ("300 goals in 334 matches") also missed. Even 200 words of that list still mixes many unrelated records. **The remaining weakness is chunk structure, not chunk size**, which points to structure-aware chunking (below) as the next improvement.

## Design decisions

- **Embed once, query many times.** The corpus is embedded once and saved; each question costs one small embedding call plus one generation call.
- **Batched embedding.** Chunks are embedded 50 per request to stay under per-request limits and minimize request count.
- **Retry with exponential backoff.** Rate limits (429) and server errors (500, 503) are retried with waits of 15s, 30s, 60s, ...; other errors (bad request, bad key) fail immediately, since retrying can't fix them. This was necessary in practice: the free tier allows 30,000 embedding tokens per minute, and the corpus is about 56,000 tokens.
- **Task-specific embeddings.** Chunks and questions are embedded with the `RETRIEVAL_DOCUMENT` and `RETRIEVAL_QUERY` task types, so the model places matching questions and passages closer together.
- **Grounded generation.** The prompt restricts the model to the retrieved sources and requires citations; questions the corpus can't answer (e.g. "What is Messi's favorite food?") return "I couldn't find that in the sources."

## Limitations

- **Tables were removed from the source documents.** Wikipedia tables don't survive copy-paste as usable text (headers separate from rows, merged cells lose labels), so facts that only appeared in tables, such as season-by-season statistics, can't be answered.
- **List-structured text retrieves poorly**, as described in the failure analysis.
- **Strict grounding can be over-cautious.** Asked "Who is Pelé?", the system retrieved chunks that mention Pelé but refused to answer, because none of them describe him directly. A partial answer would have been more useful. Low top scores (about 0.61 vs. about 0.77 for well-covered questions) are a signal of this case.
- **Small evaluation set.** 27 questions is enough to catch large effects, not small ones.
- **Generated answers vary between runs**, as is normal for LLMs. Retrieval is deterministic, which is why the evaluation measures retrieval rather than answer wording.
- **Free-tier quotas** limit throughput (e.g. about 20 generation requests per day per model at the time of writing).

## Future work

- **Structure-aware chunking**: split list-style documents by line or section, keep fixed-size windows for prose, and measure the effect on the list-based misses.
- **Hybrid retrieval**: combine embeddings with keyword search (e.g. BM25), which tends to help with exact names and numbers.
- **End-to-end answer evaluation**: score generated answers for correctness and citation accuracy, not just retrieval.
- **A larger evaluation set**, so smaller differences can be detected.
- **A single shared retry helper** for embedding and generation calls.

## Data

The documents in `data/` are adapted from English Wikipedia articles (Lionel Messi, List of career achievements by Lionel Messi, 2022 FIFA World Cup final, Messi–Ronaldo rivalry, Ballon d'Or, Inter Miami CF), used under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license. Tables and reference sections were removed.

## Background

This is a ground-up rebuild of an earlier prototype from a student club project. The original passed a whole text file to the model with no retrieval step; this version adds chunking, embeddings, vector search, grounded and cited generation, an API, and an evaluation.