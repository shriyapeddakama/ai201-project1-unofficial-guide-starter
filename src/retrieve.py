"""Milestone 4, Step 2-3: Retrieval over the ChromaDB store.

Exposes retrieve(query, k) -> list of result dicts (text, source, url,
doc_position, distance), and, when run directly, tests retrieval against the
5 evaluation questions from planning.md so we can eyeball relevance and
distance scores before wiring in generation (Milestone 5).

Distances are cosine (0 = identical .. 2 = opposite); the checkpoint target
is top results below 0.5.

Run from the project root (after `python -m src.embed`):
    python -m src.retrieve
"""

import textwrap

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    EVAL_QUERIES,
    TOP_K,
)

# Load the model and open the collection once, at import time, so repeated
# retrieve() calls (and Milestone 5) don't reload them.
_model = SentenceTransformer(EMBED_MODEL)
_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _client.get_collection(COLLECTION_NAME)


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Return the k most relevant chunks for `query`, nearest first."""
    q_emb = _model.encode([query], normalize_embeddings=True)
    res = _collection.query(query_embeddings=q_emb.tolist(), n_results=k)

    results = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        results.append({
            "text": doc,
            "source": meta["source"],
            "url": meta["url"],
            "doc_position": meta["doc_position"],
            "distance": dist,
        })
    return results


def _preview(text: str, width: int = 100, lines: int = 3) -> str:
    wrapped = textwrap.wrap(text, width=width)
    out = wrapped[:lines]
    if len(wrapped) > lines:
        out[-1] += " ..."
    return "\n      ".join(out)


def main() -> None:
    print(f"Testing retrieval (k={TOP_K}, cosine distance) against "
          f"{len(EVAL_QUERIES)} eval queries.\n")
    for i, query in enumerate(EVAL_QUERIES, 1):
        print("=" * 90)
        print(f"Q{i}: {query}")
        print("=" * 90)
        for rank, r in enumerate(results := retrieve(query), 1):
            flag = "" if r["distance"] < 0.5 else "  <-- above 0.5"
            print(f"\n  [{rank}] distance={r['distance']:.3f}{flag}  "
                  f"source: {r['source']}")
            print(f"      {_preview(r['text'])}")
        best = results[0]["distance"]
        verdict = "OK (best < 0.5)" if best < 0.5 else "WEAK (best >= 0.5)"
        print(f"\n  --> best distance {best:.3f}: {verdict}\n")


if __name__ == "__main__":
    main()
