from fastapi import FastAPI
from app.search import SemanticSearch
import os

app = FastAPI(title="Semantic Search API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "documents.txt")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    documents = f.read().splitlines()

search_engine = SemanticSearch(documents)


@app.get("/")
def root():
    return {"message": "Semantic Search API is running"}


@app.get("/search")
def search(q: str, top_k: int = 3):
    results = search_engine.search(q, top_k)

    return {
        "query": q,
        "results": [
            {
                "text": doc,
                "score": float(score)
            }
            for doc, score in results
        ]
    }