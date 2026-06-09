"""Step 4 of Milestone 3: the chunk-quality checkpoint.

Loads data/chunks.json and:
  - runs automated checks for the failure modes the milestone warns about
    (empty chunks, leftover HTML/entities/URLs, broken source metadata);
  - reports the chunk count and size distribution;
  - prints 5 random chunks so they can be read for self-containment.

Run from the project root:
    python -m src.inspect_chunks
"""

import json
import random

import numpy as np

from src.config import CHUNKS_PATH, CLEAN_DIR

SEED = 42  # fixed so the "random" sample is reproducible across runs


def main() -> None:
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    valid_sources = {
        e["source"]
        for e in json.loads((CLEAN_DIR / "manifest.json").read_text(encoding="utf-8"))
    }

    print(f"Total chunks: {len(chunks)}\n")

    # --- Automated failure-mode checks -----------------------------------
    empty = [c for c in chunks if not c["text"].strip()]
    html_like = [c for c in chunks if "<" in c["text"] and ">" in c["text"]]
    entities = [c for c in chunks if "&#" in c["text"] or "&amp;" in c["text"]
                or "&nbsp;" in c["text"]]
    urls = [c for c in chunks if "http://" in c["text"] or "https://" in c["text"]]
    bad_box = [c for c in chunks if "�" in c["text"]]
    bad_source = [c for c in chunks if c["source"] not in valid_sources]

    print("Automated checks:")
    print(f"  empty/whitespace chunks ........ {len(empty)}")
    print(f"  chunks with HTML tags .......... {len(html_like)}")
    print(f"  chunks with HTML entities ...... {len(entities)}")
    print(f"  chunks with leftover URLs ...... {len(urls)}")
    print(f"  chunks with U+FFFD box char .... {len(bad_box)}")
    print(f"  chunks with unknown source ..... {len(bad_source)}")

    # --- Size distribution ------------------------------------------------
    sizes = [c["char_count"] for c in chunks]
    print("\nChunk size (chars): "
          f"min={min(sizes)} p25={int(np.percentile(sizes, 25))} "
          f"median={int(np.median(sizes))} p75={int(np.percentile(sizes, 75))} "
          f"max={max(sizes)}")
    tiny = [c for c in chunks if c["char_count"] < 150]
    print(f"chunks under 150 chars (possible fragments): {len(tiny)}")

    # --- 5 random chunks to read -----------------------------------------
    random.seed(SEED)
    sample = random.sample(chunks, k=min(5, len(chunks)))
    print("\n" + "=" * 70)
    print("5 RANDOM CHUNKS (read these for self-containment)")
    print("=" * 70)
    for c in sample:
        print(f"\n--- chunk #{c['id']}  [{c['char_count']} chars]  "
              f"source: {c['source']} ---")
        print(c["text"])


if __name__ == "__main__":
    main()
