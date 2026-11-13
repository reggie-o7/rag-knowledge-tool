from fastapi import FastAPI
from pydantic import BaseModel

from answer import answer
from search import load_index

app = FastAPI(title="Messi RAG Knowledge Query Tool")
chunks, vectors = load_index()


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(chunks)}


@app.post("/ask")
def ask(req: AskRequest):
    return answer(req.question, chunks, vectors)
