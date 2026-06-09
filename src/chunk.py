"""Step 3 of Milestone 3: Semantic chunking.

Implements the chunking strategy from planning.md:

  - Parse each cleaned document sentence by sentence.
  - Embed every sentence with all-MiniLM-L6-v2.
  - Walk the sentences, accumulating them into a chunk. Insert a boundary when
    the semantic distance between consecutive sentences crosses a *calculated*
    threshold (the 80th percentile of that document's sentence-to-sentence
    distances) -- i.e. at the document's most significant topic shifts.
  - Keep chunks roughly 400-800 characters: a semantic break is only allowed
    once a chunk has reached MIN_CHARS, and a chunk is force-split if adding the
    next sentence would exceed MAX_CHARS.
  - Overlap: when a chunk closes, its last sentence is carried over as the first
    sentence of the next chunk, preserving continuity across the boundary.

Every chunk keeps its source metadata so retrieval can cite where it came from.

Run from the project root (first run downloads the ~80MB model):
    python -m src.chunk
"""

import json
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import CLEAN_DIR, CHUNKS_PATH

# --- Chunking parameters (from planning.md) --------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
MIN_CHARS = 400          # don't allow a semantic break before this size
MAX_CHARS = 800          # force a break before exceeding this size
DISTANCE_PERCENTILE = 80  # the "calculated threshold": break at the top 20%
                          # most significant sentence-to-sentence shifts


def split_sentences(text: str) -> list[str]:
    """Break a document into sentence-like units.

    Each non-empty line is split on sentence-ending punctuation. List items
    and headers (which have no terminal punctuation) stay intact as one unit.
    Any unit longer than MAX_CHARS is hard-split on word boundaries so a single
    runaway "sentence" can't blow past the chunk size limit.
    """
    units: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for part in re.split(r"(?<=[.!?])\s+", line):
            part = part.strip()
            if not part:
                continue
            if len(part) <= MAX_CHARS:
                units.append(part)
            else:
                units.extend(_hard_split(part, MAX_CHARS))
    return units


def _hard_split(text: str, size: int) -> list[str]:
    """Split an over-long string into <= size pieces at word boundaries."""
    words = text.split(" ")
    pieces, current = [], ""
    for w in words:
        if current and len(current) + 1 + len(w) > size:
            pieces.append(current)
            current = w
        else:
            current = f"{current} {w}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_document(sentences: list[str], embeddings: np.ndarray) -> list[str]:
    """Group sentences into ~400-800 char chunks using semantic boundaries.

    `embeddings` are L2-normalised, so cosine distance is 1 - dot product.
    """
    if not sentences:
        return []

    # The calculated threshold: distance between each adjacent sentence pair,
    # then take a high percentile so we only break at the biggest shifts.
    if len(sentences) > 1:
        adj_dist = [1.0 - float(embeddings[i] @ embeddings[i + 1])
                    for i in range(len(sentences) - 1)]
        threshold = float(np.percentile(adj_dist, DISTANCE_PERCENTILE))
    else:
        threshold = 1.0

    chunks: list[str] = []
    cur_idx: list[int] = [0]                       # sentence indices in chunk
    cur_len = len(sentences[0])

    for i in range(1, len(sentences)):
        sent = sentences[i]
        prev = cur_idx[-1]
        dist = 1.0 - float(embeddings[i] @ embeddings[prev])
        prospective = cur_len + 1 + len(sent)

        semantic_break = cur_len >= MIN_CHARS and dist > threshold
        size_break = prospective > MAX_CHARS

        if semantic_break or size_break:
            chunks.append(" ".join(sentences[j] for j in cur_idx))
            # Carry the last sentence over for continuity, unless that would
            # already overflow the next chunk on its own.
            carry = cur_idx[-1]
            if len(sentences[carry]) + 1 + len(sent) <= MAX_CHARS:
                cur_idx = [carry, i]
                cur_len = len(sentences[carry]) + 1 + len(sent)
            else:
                cur_idx = [i]
                cur_len = len(sent)
        else:
            cur_idx.append(i)
            cur_len = prospective

    chunks.append(" ".join(sentences[j] for j in cur_idx))
    return chunks


def main() -> None:
    manifest = json.loads((CLEAN_DIR / "manifest.json").read_text(encoding="utf-8"))

    print(f"Loading embedding model: {MODEL_NAME} ...", flush=True)
    model = SentenceTransformer(MODEL_NAME)

    all_chunks = []
    for entry in manifest:
        text = (CLEAN_DIR / entry["clean_file"]).read_text(encoding="utf-8")
        sentences = split_sentences(text)
        if not sentences:
            print(f"  WARNING: no sentences in {entry['source']}")
            continue

        embeddings = model.encode(sentences, normalize_embeddings=True,
                                  show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        doc_chunks = chunk_document(sentences, embeddings)
        for text_chunk in doc_chunks:
            all_chunks.append({
                "id": len(all_chunks),
                "text": text_chunk,
                "source": entry["source"],
                "type": entry["type"],
                "url": entry["url"],
                "char_count": len(text_chunk),
            })
        print(f"  {len(doc_chunks):>4} chunks  <-  {entry['source']}")

    CHUNKS_PATH.write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- Summary ----------------------------------------------------------
    n = len(all_chunks)
    sizes = [c["char_count"] for c in all_chunks]
    print(f"\nWrote {n} chunks -> {CHUNKS_PATH}")
    print(f"chunk size chars: min={min(sizes)} median={int(np.median(sizes))} "
          f"mean={int(np.mean(sizes))} max={max(sizes)}")
    if n < 50:
        print("WARNING: fewer than 50 chunks -- chunks may be too large.")
    elif n > 2000:
        print("WARNING: more than 2000 chunks -- chunks may be too small.")
    else:
        print(f"Chunk count {n} is within the healthy 50-2000 range.")


if __name__ == "__main__":
    main()
