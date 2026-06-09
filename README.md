# The Unofficial Guide — Project 1

A retrieval-augmented question-answering system for **unofficial Carnegie Mellon
campus-survival knowledge** — housing, dining, study spots, free perks, and
getting around Pittsburgh — built from student guides, official OIE pages, and
Reddit threads. Answers are generated only from the collected documents and are
shown with their source.

**How to run it:**

```bash
pip install -r requirements.txt          # install dependencies
# put your Groq key in .env  (cp .env.example .env, then edit)

python -m src.ingest        # 1. load + split sources, extract PDFs   -> data/raw/
python -m src.clean         # 2. clean each document                  -> data/clean/
python -m src.chunk         # 3. semantic chunking                    -> data/chunks.json
python -m src.embed         # 4. embed + load into ChromaDB           -> chroma_db/
python -m src.evaluate      # run the 5 eval questions end-to-end
python app.py               # 5. launch the Gradio UI at localhost:7860
```

---

## Domain

This system covers the **unofficial "campus survival" knowledge** that
incoming and current Carnegie Mellon students rely on but that is scattered
across student-run wikis and Reddit rather than official channels: how meal
plans and meal blocks actually work, which dorms/neighborhoods people live in,
where to study, what you get for free with your Andrew ID, and how to get around
Pittsburgh on transit.

This knowledge is valuable because CMU's official settling-in guides explain
*policy* (deadlines, applications, where to submit forms) but rarely the
*lived experience* — which the unofficial sources capture candidly (e.g. "buy
the red meal plan, then switch to yellow if you get an apartment dorm," or "move
off campus ASAP — nicer apartments, cheaper rent"). It is hard to find through
official channels because it is opinion- and experience-based, frequently
out of date on any single page, and spread across many Reddit threads and a
community wiki that a newcomer wouldn't know to search.

---

## Document Sources

10 sources spanning a student wiki (cmu.guide), official CMU/OIE settling-in
pages, two PDF guides, and four Reddit r/cmu threads. The eight web/Reddit
sources arrived merged into a single `Compiled Sources.txt`; the ingestion step
splits them back into separate logical documents so every chunk keeps a
meaningful source label (see `src/config.py`).

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | CMU Guide: Mooching off CMU (free stuff) | web (wiki) | https://cmu.guide/MoochingOffCMU/ |
| 2 | CMU OIE: Graduate Student Checklist | web (official) | https://www.cmu.edu/oie/pre-arrival-and-settling-in/settling-in-guide/your-first-weeks/graduate-checklist.html |
| 3 | CMU OIE: Housing | web (official) | https://www.cmu.edu/oie/pre-arrival-and-settling-in/settling-in-guide/housing.html |
| 4 | CMU Guide: Meal Plans | web (wiki) | https://cmu.guide/meal-plans/ |
| 5 | Reddit r/cmu: Help me fall in love with CMU | reddit | https://www.reddit.com/r/cmu/comments/1s54fio/help_me_fall_in_love_with_cmu/ |
| 6 | Reddit r/cmu: Best study spots on campus | reddit | https://www.reddit.com/r/cmu/comments/1j07hl1/best_spots_on_campus/ |
| 7 | Reddit r/cmu: Everything about CMU housing | reddit | https://www.reddit.com/r/cmu/comments/16y9hn/reference_thread_everything_you_ever_wanted_to/ |
| 8 | Reddit r/cmu: What to do on campus and in Pittsburgh | reddit | https://www.reddit.com/r/cmu/comments/1db47r/reference_thread_everything_you_ever_wanted_to/ |
| 9 | CMU Graduate Student Handbook | pdf | `documents/Graduate Student Handbook.pdf` |
| 10 | Tepper Survival Guide | pdf | `documents/Tepper Guide.pdf` |

---

## Chunking Strategy

**Chunk size:** Dynamic, targeting **400–800 characters** per chunk (median in
practice: ~600).

**Overlap:** **1-sentence carry-over.** When a chunk closes, its final sentence
becomes the first sentence of the next chunk, so a thought that spans a boundary
isn't lost to either side.

**Preprocessing before chunking** (`src/clean.py`), applied per source type:
- *All sources:* decode HTML entities, strip HTML tags and URLs, normalize
  whitespace. The web/Reddit text was corrupted with `U+FFFD` replacement
  characters where apostrophes, dashes, and bullets used to be — these are
  restored to apostrophes only when followed by a real contraction suffix
  (`'s`, `'re`, `n't`, …) and otherwise replaced with a space.
- *Web/Reddit:* remove wiki boilerplate (`Edit page on GitHub`, `Last edited:`,
  `Contributors:`).
- *PDF:* drop table-of-contents lines (dotted leaders + page number), remove
  repeated running headers/footers (detected by frequency after stripping page
  numbers), fix pdfplumber's duplicated title glyphs (`SSSSUUUU…` → `SURVIVAL`),
  and **reflow hard line-wraps** so sentences broken across PDF lines are
  rejoined before sentence-splitting.

**Why these choices fit the documents:** The corpus mixes long multi-page PDF
guides with short, punchy Reddit comments. A fixed character cut would slice
mid-sentence and split a single piece of advice from its context. Instead the
chunker (`src/chunk.py`) splits each document into sentences, embeds every
sentence with `all-MiniLM-L6-v2`, and inserts a boundary at the document's
largest topic shifts — specifically where the cosine distance between adjacent
sentences exceeds that document's **80th-percentile** distance — but only once a
chunk has reached 400 characters, and it force-breaks before 800. This keeps
chunks self-contained (a complete idea, e.g. one Reddit comment or one
meal-plan rule) while bounding their size so each embedding carries focused
meaning.

**Final chunk count:** **377 chunks** across the 10 documents (Tepper Guide 219,
Handbook 81, the rest 6–16 each). This sits comfortably inside the healthy
50–2,000 range; automated checks confirm 0 empty chunks, 0 HTML/entity/URL
leftovers, and 0 fragments under 150 characters.

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`, with embeddings
stored in **ChromaDB** using **cosine** distance. It runs locally with no API
key or rate limits, is fast on CPU, and produced strong retrieval distances
(top results 0.32–0.48 across all five eval queries). Retrieval uses **top-k =
5**.

**Production tradeoff reflection:** For a real, university-wide deployment with
no cost constraint, I would reconsider `all-MiniLM-L6-v2` mainly because of its
**256-token context window**, which truncates the longer PDF-derived chunks and
loses signal. The bigger risk is **domain-specific jargon**: CMU students rely
on terms like *CUC, Andrew ID, the HUB, Tepper, b-boards, flex dollars,* and bus
route codes, and a general-purpose model can blur these against unrelated text.
In production I'd weigh (a) a larger-context, higher-accuracy embedding model
(e.g. an OpenAI or BGE/E5-large model) for better semantic recall on long
chunks, against (b) added **hybrid search** — combining dense embeddings with a
keyword/BM25 channel — so exact tokens like "CUC", "71B", or "Sorrells" are
matched reliably even when the dense model ranks them low (a failure I actually
observed; see below). The tradeoffs are latency and cost (API-hosted models add
both) versus accuracy on this jargon-heavy, place-name-heavy domain.

---

## Grounded Generation

Generation uses **Groq's `llama-3.3-70b-versatile`** at `temperature=0`
(`src/generate.py`). Retrieved chunks are formatted into a numbered,
source-labelled context block and passed with the question.

**System prompt grounding instruction** (verbatim from `src/generate.py`):

> You are the CMU Unofficial Guide… You must answer using ONLY the information
> in the numbered context documents provided in the user's message.
> 1. Do NOT use any outside or prior knowledge. If a fact is not stated in the
>    context, you do not know it.
> 2. If the context does not contain enough information to answer the question,
>    reply with exactly this sentence and nothing else: "I don't have enough
>    information on that."
> 3. Do not invent names, numbers, prices, bus routes, or places that are not
>    present in the context.
> 4. When you do answer, cite the supporting context items inline using their
>    bracketed numbers, e.g. [1] or [2].

This *enforces* grounding rather than suggesting it: it forbids prior knowledge,
mandates an exact refusal string, and `temperature=0` makes the behavior
deterministic. Tested with out-of-scope questions ("What is the weather in
Paris?", "How do I file my taxes in Germany?"), the system returns exactly *"I
don't have enough information on that."* with no sources.

**How source attribution is surfaced:** Attribution is built **programmatically
in code**, not left to the model. After generation, `ask()` collects the unique
`source` metadata of the retrieved chunks (in rank order) and returns them as a
`sources` list that the UI shows under "Retrieved from". Because this comes from
the retrieval metadata rather than the LLM's text, sources can't be
hallucinated. When the model declines to answer, the sources list is emptied so
the UI doesn't imply a grounded answer where there wasn't one.

---

## Evaluation Report

Run with `python -m src.evaluate` (k=5, cosine distance). All five top
retrievals scored below 0.5, so retrieval quality was at least *partially
relevant* on every question; accuracy varied with whether the *specific*
expected facts existed in the corpus and were ranked into the top-5.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Which bus lines from Oakland campus to Shadyside grocery stores? | 75 or 71B | Lists 71A/71C/71D/500/78C and 61-series toward Downtown, hedging on the exact route (best dist 0.320) | Relevant | Partially accurate |
| 2 | What can students do free with their Andrew ID? | Museums, Arts Pass, free buses, subscriptions | Museums, CFA/Music performances, NYT/WSJ/GitHub/Overleaf subscriptions — but **omits the free PRT bus pass** (best dist 0.480) | Relevant | Partially accurate |
| 3 | Good study spots around campus? | Hunt, Sorrells libraries | Hunt, Gates, Tepper, Scaife, ANSYS, Donner, Doherty, Cohon Center — Hunt correct; **Sorrells not surfaced** (best dist 0.319) | Relevant | Accurate |
| 4 | Good on-campus vs off-campus dining? | On: Scottys/Stackd/Millies. Off: Noodlehead/Senyai/Shah's | Correct on/off split; Millie's (on-campus); Eat Unique, Bao, Mercurio's, Pamela's (off-campus) — but the **specific expected places aren't in the corpus** (best dist 0.320) | Partially relevant | Partially accurate |
| 5 | Housing options for first-year graduate students? | Shadyside, Oakland, Squirrel Hill; no dorms for grad students | Correctly says grad students get **no dorms / off-campus only**, but **omits the three neighborhoods** (best dist 0.356) | Relevant | Partially accurate |

**Overall:** Retrieval consistently pulled on-topic chunks from the right
sources. The recurring limitation was *recall and coverage*, not topical
relevance: answers were correct as far as the retrieved context went, but
sometimes incomplete because a specific expected fact was either absent from the
collected documents (Q1 `71B`, Q4 Noodlehead/Senyai) or present but ranked
outside the top-5 (Q3 Sorrells, Q5 neighborhoods).

---

## Failure Case Analysis

**Question that failed:** Q5 — "What housing options do first year graduate
students have?" (Expected: Shadyside, Oakland, Squirrel Hill; no dorms for grad
students.)

**What the system returned:** "Graduate students, including first-year graduate
students, should expect to arrange their own housing at off-campus locations …
on CMU's Off-Campus Housing website." This is correct on the key point (no
dorms; off-campus only) but **never names the three neighborhoods**, which were
the most concrete part of the expected answer.

**Root cause (tied to a specific pipeline stage):** This is a **retrieval recall
failure** rooted in the **embedding + chunking** stages. The neighborhood facts
live in a single chunk of *CMU OIE: Housing* that reads "The following
neighborhoods are located within a two-mile radius of campus … Oakland,
Shadyside and Squirrel Hill: Most CMU students choose to live in these three
areas." That chunk is *about proximity and neighborhoods*, so its embedding sits
near concepts like "two-mile radius" and "neighborhoods," not near the query's
emphasis on *"graduate," "housing options," "first year."* I confirmed this by
retrieving the top 12 chunks for the query: the neighborhoods chunk **does not
appear even at rank 12** — it loses to the OIE housing intro and the Reddit
housing-lottery chunks, which match the query's framing more directly. With
k=5, generation never receives the neighborhoods chunk, so the model — correctly
refusing to invent place names — answers accurately but incompletely.

**What I would change to fix it:** Three options, in increasing effort:
(1) raise k (e.g. to 8–10) so more of the *CMU OIE: Housing* chunks reach the
model — cheap, but dilutes context on other queries; (2) add **hybrid search**
so a query mentioning "housing"/"live" also keyword-matches the neighborhood
chunk; (3) **re-chunk** the housing document so the "graduate students arrange
off-campus housing" statement and the neighborhood list co-occur in one chunk,
giving the combined idea a single, query-aligned embedding. Option (3) addresses
the real cause — semantic chunking separated two facts that belong together for
this question.

---

## Spec Reflection

**One way the spec helped me during implementation:** Writing the **Documents**
and **Chunking Strategy** sections of `planning.md` first turned vague intent
into concrete code targets. The chunking spec ("400–800 characters, semantic
boundaries via sentence embeddings, carry sentences across breaks") translated
almost directly into `chunk.py`'s parameters and control flow, and having the
five **evaluation questions** written in advance meant I could test retrieval
the moment the vector store existed — I built `EVAL_QUERIES` into `config.py` and
reused the same questions for the M4 retrieval check and the M6 report. The spec
also exposed a data problem early: it listed 10 distinct sources, but on disk the
8 web/Reddit ones were merged into one file, which pushed me to split them back
apart on ingestion so attribution would actually work.

**One way my implementation diverged from the spec, and why:** The plan
described chunking as "insert a chunk boundary when the semantic distance between
sentences crosses a calculated threshold." A *pure* semantic threshold ignores
size, which produced both 2,000-character mega-chunks (long stretches with no
sharp topic shift) and one-sentence fragments. So I **combined** the semantic
threshold with hard size bounds: a boundary is only allowed after 400 characters
and is forced before 800, and the "calculated threshold" became each document's
80th-percentile adjacent-sentence distance rather than a single global constant.
I also made the "adaptive overlap" concrete as a fixed 1-sentence carry-over,
since the original wording wasn't precise enough to implement. These changes kept
the spec's *intent* (boundaries at meaning shifts) while making chunk sizes
predictable for retrieval.

---

## AI Usage

**Instance 1 — Ingestion, cleaning, and chunking pipeline**

- *What I gave the AI:* My `planning.md` Domain, Documents, and Chunking Strategy
  sections, plus the actual files in `documents/` (the merged
  `Compiled Sources.txt` and the two PDFs), and asked it to build a pipeline that
  loads, cleans, and semantically chunks them.
- *What it produced:* A three-stage `src/` pipeline (`ingest.py`, `clean.py`,
  `chunk.py`) implementing the 400–800-char semantic chunker.
- *What I changed or overrode:* I directed it to **split the merged
  `Compiled Sources.txt` back into 8 logical sources** during ingestion (so
  chunks carry meaningful source labels) rather than treating it as one document.
  When inspecting cleaned output I caught two cleaning bugs and had it fix them:
  an over-eager rule that turned separators into apostrophes (`Office'on` →
  `Office on`), and PDF sentences broken across hard line-wraps, which I had it
  solve by adding a reflow step. I also confirmed the `�` boxes were `U+FFFD`
  corruption (unrecoverable) and chose to normalize rather than try to restore
  the original characters.

**Instance 2 — Embedding, retrieval, and grounded generation**

- *What I gave the AI:* My `planning.md` Retrieval Approach section and the
  architecture diagram, asking for the ChromaDB embedding/retrieval code and a
  grounded generation function over Groq.
- *What it produced:* `embed.py` (embed + load into ChromaDB), `retrieve.py`
  (`retrieve(query, k=5)`), and `generate.py` (`ask()` calling
  `llama-3.3-70b-versatile`).
- *What I changed or overrode:* I directed it to configure ChromaDB for
  **cosine** distance (so scores read 0–2 and the <0.5 checkpoint is meaningful)
  and to make **source attribution programmatic** — collected from retrieval
  metadata in code — instead of trusting the LLM to cite sources in its text,
  which could be hallucinated. I set generation `temperature=0` and tightened the
  system prompt to mandate the exact "I don't have enough information on that."
  refusal and to forbid inventing bus routes/place names, after deciding
  grounding needed to be enforced rather than suggested.
