# ingest_games.py
import os
import json
import glob
import weaviate
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
CLASS_NAME = os.getenv("WEAVIATE_CLASS", "GameKnowledge")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

client = weaviate.Client(url=WEAVIATE_URL)
model = SentenceTransformer(f"sentence-transformers/{EMBED_MODEL}")

def create_schema():
    # delete if exists
    try:
        classes = client.schema.get("classes")
        for c in classes:
            if c.get("class") == CLASS_NAME:
                client.schema.delete_class(CLASS_NAME)
                break
    except Exception:
        pass

    class_schema = {
        "class": CLASS_NAME,
        "vectorizer": "none",  # we'll provide vectors
        "properties": [
            {"name": "game", "dataType": ["text"]},
            {"name": "genre", "dataType": ["text"]},
            {"name": "section", "dataType": ["text"]},
            {"name": "content", "dataType": ["text"]},
        ],
    }
    client.schema.create_class(class_schema)
    print("Created schema:", CLASS_NAME)

def load_docs():
    # priority: data/games/*.md
    docs = []
    md_files = glob.glob("data/games/*.md")
    if md_files:
        for p in md_files:
            with open(p, "r", encoding="utf-8") as f:
                text = f.read().strip()
            title = os.path.splitext(os.path.basename(p))[0]
            docs.append({"game": title, "genre": "", "section": "full", "content": text})
        return docs

    # else try data/games_data.json
    if os.path.exists("data/games_data.json"):
        with open("data/games_data.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        # expected format: list of { "game": "...", "section": "...", "content": "...", "genre": "..." }
        for item in raw:
            docs.append({
                "game": item.get("game") or item.get("title") or "Unknown",
                "genre": item.get("genre", ""),
                "section": item.get("section", ""),
                "content": item.get("content", "")
            })
        return docs

    raise SystemExit("No docs found. Place markdowns in data/games/ or create data/games_data.json")

def chunk_text(text, chunk_size=450, overlap=80):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def ingest():
    docs = load_docs()
    for doc in tqdm(docs, desc="Ingesting docs"):
        chunks = chunk_text(doc["content"])
        for i, c in enumerate(chunks):
            vector = model.encode(f"{doc['game']}\n\n{c}").astype(np.float32).tolist()
            obj = {
                "game": doc["game"],
                "genre": doc.get("genre", ""),
                "section": doc.get("section", ""),
                "content": c,
            }
            client.data_object.create(data_object=obj, class_name=CLASS_NAME, vector=vector)

if __name__ == "__main__":
    create_schema()
    ingest()
    print("Done.")
