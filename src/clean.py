"""Step 2 of Milestone 3: Cleaning.

Read each raw file from data/raw/, strip everything that isn't substantive
content, and write the result to data/clean/. Cleaning rules are split into:

  - clean_common(): rules every document needs (HTML entities/tags, URLs,
    the U+FFFD `?` corruption, whitespace).
  - clean_web():    boilerplate specific to the scraped web/reddit pages
    ("Edit page on GitHub", "Last edited:", "Contributors:").
  - clean_pdf():    PDF extraction junk (table-of-contents dotted leaders,
    repeated page footers, the quadrupled-glyph title, page-break markers).

Which rules run is decided by the "type" field in data/raw/manifest.json.

Run from the project root:
    python -m src.clean
"""

import html
import json
import re

from src.config import RAW_DIR, CLEAN_DIR


# --- Rules shared by every document ---------------------------------------

def clean_common(text: str) -> str:
    # Decode HTML entities (&amp; -> &, &#39; -> ', &nbsp; -> space) and drop
    # any stray HTML tags. The scraped text is mostly tag-free already, but
    # this guarantees no leftover markup reaches the chunker.
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs. They are noise for an embedding model and add no meaning to
    # a survival-guide answer. We stop the match at whitespace OR at the
    # U+FFFD corruption character, because in this corpus URLs are often glued
    # directly to the following sentence by a U+FFFD with no space.
    text = re.sub(r"https?://[^\s�]+", " ", text)
    text = re.sub(r"www\.[^\s�]+", " ", text)

    # Handle the U+FFFD replacement character (the corrupted box glyph):
    #   1. A letter, then U+FFFD, then a real English contraction suffix
    #      (s, re, ve, ll, d, m, t) was almost always an apostrophe:
    #      "they<U+FFFD>re" -> "they're", "CMU<U+FFFD>s" -> "CMU's",
    #      "don<U+FFFD>t" -> "don't". Requiring a known suffix avoids turning
    #      separators into apostrophes, e.g. "Housing Office<U+FFFD>on finding"
    #      stays "Housing Office on finding" rather than "Office'on".
    text = re.sub(r"(?<=[A-Za-z])�(?=(?:s|re|ve|ll|d|m|t)\b)", "'", text)
    #   2. Anything left (runs of it, or it sitting next to punctuation/spaces/
    #      capitals) was a bullet, dash, emoji, or separator we can't recover.
    text = re.sub(r"�+", " ", text)

    # Normalise whitespace: collapse runs of spaces/tabs, trim each line, and
    # collapse 3+ blank lines down to a single blank line.
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- Web / reddit boilerplate ---------------------------------------------

# Lines that appear on the cmu.guide pages and carry no domain content.
_WEB_BOILERPLATE = re.compile(
    r"^(Edit page on GitHub|Last edited:.*|Contributors:.*)$",
    re.IGNORECASE,
)


def clean_web(text: str) -> str:
    kept = [ln for ln in text.splitlines() if not _WEB_BOILERPLATE.match(ln.strip())]
    return "\n".join(kept)


# --- PDF extraction junk ---------------------------------------------------

# A table-of-contents line: text, then dotted leaders, then a page number,
# e.g. "1.5: Academic Calendar...............7"
_TOC_LINE = re.compile(r"\.{2,}\s*\d+\s*$")
# A bare page-number line.
_PAGE_NUM = re.compile(r"^\d+$")
# Strip a leading or trailing standalone page number so that running
# headers/footers like "2 Prepared by ... Club" and "Prepared by ... Club 3"
# normalise to the same string and can be counted together.
_EDGE_PAGE_NUM = re.compile(r"^\d+\s+|\s+\d+$")


def _repeated_boilerplate(lines: list[str], min_repeats: int = 5, max_len: int = 80) -> set[str]:
    """Find short lines that recur many times across the document.

    Running headers and footers (e.g. "Office of Graduate & Postdoctoral
    Affairs <page>") repeat on every page. After stripping the page number,
    any short line appearing >= min_repeats times is treated as boilerplate.
    Returns the set of normalised forms to drop.
    """
    counts: dict[str, int] = {}
    for line in lines:
        norm = _EDGE_PAGE_NUM.sub("", line.strip()).strip()
        if norm and len(norm) <= max_len:
            counts[norm] = counts.get(norm, 0) + 1
    return {norm for norm, c in counts.items() if c >= min_repeats}


_ENDS_SENTENCE = re.compile(r"""[.!?:]["')’]?$""")


def _reflow(lines: list[str]) -> list[str]:
    """Re-join PDF lines that were hard-wrapped mid-sentence.

    pdfplumber keeps the PDF's visual line breaks, so one sentence can span
    several lines. We join a line onto the previous one when the previous line
    does NOT end with sentence punctuation AND this line continues in lowercase
    -- headings, bullets and new sentences start with an uppercase letter or a
    marker, so those line breaks are preserved.
    """
    out: list[str] = []
    for line in lines:
        s = line.strip()
        if (out and out[-1] and s and s[0].islower()
                and not _ENDS_SENTENCE.search(out[-1])):
            out[-1] = f"{out[-1]} {s}"
        else:
            out.append(s)
    return out


def clean_pdf(text: str) -> str:
    # Drop the form-feed page markers we inserted during ingestion.
    text = text.replace("\f", "\n")

    # Fix glyphs that pdfplumber duplicated on stylized text, e.g. the title
    # "SSSSUUUURRRRVVVVIIIIVVVVAAAALLLL" -> "SURVIVAL". Only collapse runs of
    # 4+ identical *letters*, which effectively never occur in real words.
    text = re.sub(r"([A-Za-z])\1{3,}", r"\1", text)

    lines = text.splitlines()
    boilerplate = _repeated_boilerplate(lines)

    kept = []
    for line in lines:
        stripped = line.strip()
        if _TOC_LINE.search(stripped):      # table-of-contents entry
            continue
        if _PAGE_NUM.match(stripped):       # lone page number
            continue
        norm = _EDGE_PAGE_NUM.sub("", stripped).strip()
        if norm in boilerplate:             # repeated running header/footer
            continue
        kept.append(line)

    # Re-join sentences broken across the PDF's hard line wraps.
    return "\n".join(_reflow(kept))


# --- Dispatch --------------------------------------------------------------

def clean_document(text: str, doc_type: str) -> str:
    """Apply the cleaning pipeline appropriate for a document's type."""
    if doc_type == "pdf":
        text = clean_pdf(text)
    else:  # web, reddit
        text = clean_web(text)
    # clean_common runs last so whitespace introduced by the steps above
    # (e.g. removed lines, stripped URLs) gets normalised away.
    return clean_common(text)


def main() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RAW_DIR / "manifest.json").read_text(encoding="utf-8"))

    clean_manifest = []
    for entry in manifest:
        raw = (RAW_DIR / entry["raw_file"]).read_text(encoding="utf-8")
        cleaned = clean_document(raw, entry["type"])

        out_name = entry["raw_file"]
        (CLEAN_DIR / out_name).write_text(cleaned, encoding="utf-8")

        clean_manifest.append({
            "source": entry["source"],
            "type": entry["type"],
            "url": entry["url"],
            "clean_file": out_name,
            "raw_chars": entry["char_count"],
            "clean_chars": len(cleaned),
        })

    (CLEAN_DIR / "manifest.json").write_text(
        json.dumps(clean_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- Summary: how much each document shrank --------------------------
    print(f"Cleaned {len(clean_manifest)} documents -> {CLEAN_DIR}\n")
    print(f"{'raw':>8} {'clean':>8} {'kept':>6}  source")
    print("-" * 64)
    for e in clean_manifest:
        pct = 100 * e["clean_chars"] / e["raw_chars"] if e["raw_chars"] else 0
        print(f"{e['raw_chars']:>8} {e['clean_chars']:>8} {pct:>5.0f}%  {e['source']}")


if __name__ == "__main__":
    main()
