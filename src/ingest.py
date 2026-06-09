"""Step 1 of Milestone 3: Ingestion.

Load every document from disk and save the RAW (uncleaned) text to a
consistent format: one UTF-8 .txt file per logical source in data/raw/,
plus a manifest.json recording each source's metadata.

We deliberately do NOT clean anything here. Saving the raw text first means
we can re-run cleaning and chunking later without re-reading the PDFs, and we
can diff raw-vs-clean to confirm cleaning actually did something (Step 2).

Run from the project root:
    python -m src.ingest
"""

import json

import pdfplumber

from src.config import (
    COMPILED_FILE,
    COMPILED_SOURCES,
    PDF_SOURCES,
    DOCUMENTS_DIR,
    RAW_DIR,
    slugify,
)


def split_compiled_file() -> list[dict]:
    """Split Compiled Sources.txt into its logical sources.

    Each source begins with a unique header line listed in COMPILED_SOURCES.
    We walk the file line by line, find each anchor in order, and slice the
    text between consecutive anchors. Matching is exact (stripped) full-line
    equality, so a generic word like 'Housing' only matches its own header
    line and not the word appearing mid-paragraph.
    """
    # Read with errors="replace" so any stray bad bytes become U+FFFD rather
    # than crashing; Step 2 (cleaning) strips those out.
    text = COMPILED_FILE.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Find the line index where each anchor starts.
    anchors = [s["anchor"] for s in COMPILED_SOURCES]
    start_indices = []
    search_from = 0
    for anchor in anchors:
        found = None
        for i in range(search_from, len(lines)):
            if lines[i].strip() == anchor:
                found = i
                break
        if found is None:
            raise ValueError(
                f"Could not find header line for source anchor: {anchor!r}. "
                "The compiled file may have changed — update COMPILED_SOURCES."
            )
        start_indices.append(found)
        search_from = found + 1

    # Slice each source from its anchor up to the next anchor.
    docs = []
    for n, meta in enumerate(COMPILED_SOURCES):
        start = start_indices[n]
        end = start_indices[n + 1] if n + 1 < len(start_indices) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        docs.append({**meta, "raw_text": body})
    return docs


def load_pdfs() -> list[dict]:
    """Extract raw text from each PDF, page by page, using pdfplumber.

    Pages are joined with a form-feed marker so we can still tell where page
    breaks were if we need to during debugging.
    """
    docs = []
    for meta in PDF_SOURCES:
        path = DOCUMENTS_DIR / meta["file"]
        print(f"  reading PDF: {meta['file']} ...", flush=True)
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        body = "\n\f\n".join(pages).strip()
        docs.append({**meta, "raw_text": body, "n_pages": len(pages)})
    return docs


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Splitting Compiled Sources.txt into logical sources...")
    compiled_docs = split_compiled_file()
    print("Extracting PDFs...")
    pdf_docs = load_pdfs()

    all_docs = compiled_docs + pdf_docs

    manifest = []
    for doc in all_docs:
        slug = slugify(doc["source"])
        out_path = RAW_DIR / f"{slug}.txt"
        out_path.write_text(doc["raw_text"], encoding="utf-8")
        entry = {
            "source": doc["source"],
            "type": doc["type"],
            "url": doc["url"],
            "raw_file": out_path.name,
            "char_count": len(doc["raw_text"]),
        }
        if "n_pages" in doc:
            entry["n_pages"] = doc["n_pages"]
        manifest.append(entry)

    (RAW_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- Summary so we can eyeball that ingestion worked ------------------
    print("\nIngested {} documents -> {}".format(len(manifest), RAW_DIR))
    print(f"{'chars':>8}  {'type':<7}  source")
    print("-" * 60)
    for e in manifest:
        print(f"{e['char_count']:>8}  {e['type']:<7}  {e['source']}")
    total = sum(e["char_count"] for e in manifest)
    print("-" * 60)
    print(f"{total:>8}  total characters across {len(manifest)} sources")


if __name__ == "__main__":
    main()
