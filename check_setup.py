import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()           # loads GEMINI_API_KEY from .env
client = genai.Client() # finds the key automatically

texts = [
    "What do I need to take before CMPSC 461?",                       # the question
    "CMPSC 461 requires CMPSC 221 and CMPSC 360 as prerequisites.",   # relevant chunk
    "The Nittany Lion Shrine was built in 1942.",                     # irrelevant chunk
]

result = client.models.embed_content(model="gemini-embedding-001", contents=texts)
vectors = [np.array(e.values) for e in result.embeddings]

print("Vector length:", len(vectors[0]))


def cosine_similarity(a, b):
    return (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# print(cosine_similarity(np.array([3, 4]), np.array([4, 3])))


q = vectors[0]
print("Question vs relevant chunk:  ", cosine_similarity(q, vectors[1]))
print("Question vs irrelevant chunk:", cosine_similarity(q, vectors[2]))