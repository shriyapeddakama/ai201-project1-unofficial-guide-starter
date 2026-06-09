"""Shared paths and the source registry for the Unofficial Guide pipeline.

Everything that the ingestion / cleaning / chunking steps need to agree on
lives here so there is a single source of truth.
"""

from pathlib import Path

# --- Project paths ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"        # original source files
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                         # Step 1 output: raw text
CLEAN_DIR = DATA_DIR / "clean"                     # Step 2 output: cleaned text
CHUNKS_PATH = DATA_DIR / "chunks.json"             # Step 3 output: final chunks

# The single file that holds the 8 scraped web + reddit sources, mashed
# together with no clean delimiter. We split it back apart on ingest.
COMPILED_FILE = DOCUMENTS_DIR / "Compiled Sources.txt"

# --- Source registry -------------------------------------------------------
# Each scraped source inside Compiled Sources.txt begins with a unique header
# line. We split the file on these exact lines (matched in order) so every
# chunk can carry a meaningful source label instead of all 8 collapsing into
# "Compiled Sources.txt". Order matters: anchors are matched top to bottom.
COMPILED_SOURCES = [
    {
        "anchor": "Mooching off of CMU",
        "source": "CMU Guide: Mooching off CMU (free stuff)",
        "type": "web",
        "url": "https://cmu.guide/MoochingOffCMU/",
    },
    {
        "anchor": "Graduate Student Checklist",
        "source": "CMU OIE: Graduate Student Checklist",
        "type": "web",
        "url": "https://www.cmu.edu/oie/pre-arrival-and-settling-in/settling-in-guide/your-first-weeks/graduate-checklist.html",
    },
    {
        "anchor": "Housing",
        "source": "CMU OIE: Housing",
        "type": "web",
        "url": "https://www.cmu.edu/oie/pre-arrival-and-settling-in/settling-in-guide/housing.html",
    },
    {
        "anchor": "Meal Plans",
        "source": "CMU Guide: Meal Plans",
        "type": "web",
        "url": "https://cmu.guide/meal-plans/",
    },
    {
        "anchor": "Help Me Fall in Love With CMU",
        "source": "Reddit r/cmu: Help me fall in love with CMU",
        "type": "reddit",
        "url": "https://www.reddit.com/r/cmu/comments/1s54fio/help_me_fall_in_love_with_cmu/",
    },
    {
        "anchor": "Best Spots on Campus?",
        "source": "Reddit r/cmu: Best study spots on campus",
        "type": "reddit",
        "url": "https://www.reddit.com/r/cmu/comments/1j07hl1/best_spots_on_campus/",
    },
    {
        "anchor": "Everything You Ever Wanted to Know About CMU Housing, On and Off Campus",
        "source": "Reddit r/cmu: Everything about CMU housing",
        "type": "reddit",
        "url": "https://www.reddit.com/r/cmu/comments/16y9hn/reference_thread_everything_you_ever_wanted_to/",
    },
    {
        "anchor": "Everything You Ever Wanted to Know About What to Do on Campus and in Pittsburgh",
        "source": "Reddit r/cmu: What to do on campus and in Pittsburgh",
        "type": "reddit",
        "url": "https://www.reddit.com/r/cmu/comments/1db47r/reference_thread_everything_you_ever_wanted_to/",
    },
]

# The two PDF guides, loaded with pdfplumber.
PDF_SOURCES = [
    {
        "file": "Graduate Student Handbook.pdf",
        "source": "CMU Graduate Student Handbook",
        "type": "pdf",
        "url": "Graduate Student Handbook.pdf",
    },
    {
        "file": "Tepper Guide.pdf",
        "source": "Tepper Survival Guide",
        "type": "pdf",
        "url": "Tepper Guide.pdf",
    },
]


def slugify(text: str) -> str:
    """Turn a source name into a safe filename stem, e.g. 'CMU OIE: Housing'
    -> 'cmu-oie-housing'."""
    keep = [c.lower() if c.isalnum() else "-" for c in text]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")
