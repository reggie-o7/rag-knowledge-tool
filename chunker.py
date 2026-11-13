import re
from pathlib import Path

DATA_DIR = Path("data")
CHUNK_SIZE = 300
OVERLAP = 50


def load_documents(data_dir):
    """Return a list of (filename, text) pairs, one per .txt file."""
    pairs = []
    for path in data_dir.glob("*.txt"):
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
            pairs.append((path.name, content))
    return pairs


def clean_text(text):
    """Remove Wikipedia citation markers like [25], [B], [note 1]."""
    cleaned = re.sub(r"\[.*?\]", "", text)
    return cleaned


def chunk_text(text, chunk_size, overlap):
    """Split text into overlapping chunks of ~chunk_size words. Return a list of strings."""
    chunks = []
    words = text.split()
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        piece = words[i : i + chunk_size]
        chunks.append(" ".join(piece))
        if i + chunk_size >= len(words):
            break
    return chunks


def build_chunks():
    """Load, clean, and chunk every document. Return a list of dicts."""
    chunks = []
    for filename, text in load_documents(DATA_DIR):
        cleaned = clean_text(text)
        for i, piece in enumerate(chunk_text(cleaned, CHUNK_SIZE, OVERLAP)):
            chunks.append(
                {
                    "id": f"{filename}#{i}",  # e.g. "messi_main.txt#4"
                    "source": filename,
                    "text": piece,
                }
            )
    return chunks


if __name__ == "__main__":
    chunks = build_chunks()
