"""Milestone 4, Step 1: Embed chunks and load them into ChromaDB.

Reads data/chunks.json (the Milestone 3 output), embeds every chunk with
all-MiniLM-L6-v2, and stores them in a persistent ChromaDB collection along
with metadata (source name, type, url, and the chunk's position within its
source document) for later attribution.

The collection is configured for COSINE distance, so retrieval distances run
0 (identical) .. 2 (opposite); good matches land well under 0.5.

Run from the project root:
    python -m src.embed
"""

import json

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    CHUNKS_PATH,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
)


def add_doc_positions(chunks: list[dict]) -> list[dict]:
    """Tag each chunk with its position within its own source document.

    chunks.json is already in document order, so we just count per source.
    """
    seen: dict[str, int] = {}
    for c in chunks:
        pos = seen.get(c["source"], 0)
        c["doc_position"] = pos
        seen[c["source"]] = pos + 1
    return chunks


def main() -> None:
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    chunks = add_doc_positions(chunks)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH.name}")

    # Embed all chunk texts. normalize_embeddings=True pairs with cosine space.
    print(f"Embedding with {EMBED_MODEL} ...", flush=True)
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Persistent store so we don't have to re-embed every run. Rebuild the
    # collection from scratch each time to avoid stale/duplicate entries.
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # didn't exist yet
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[str(c["id"]) for c in chunks],
        embeddings=embeddings.tolist(),
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "source": c["source"],
                "type": c["type"],
                "url": c["url"],
                "doc_position": c["doc_position"],
            }
            for c in chunks
        ],
    )

    print(f"\nStored {collection.count()} chunks in ChromaDB collection "
          f"'{COLLECTION_NAME}' at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
