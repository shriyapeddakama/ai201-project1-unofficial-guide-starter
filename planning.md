# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->

---
Campus Survival Guide for Carnegie Mellon University focused on topics like housing, dining, things to do around Pittsburgh and advice from students on alumini to better handle CMU life. 
Althoug CMU provides settling in guides a lot of  useful knowledge is passed through unofficial channels like reddit. For examples, how to purchase meal blocks, or good study spots on campus.

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | cmu.guide| freebies you get with id| https://cmu.guide/MoochingOffCMU/ |
| 2 | cmu | graduate student checklist |https://www.cmu.edu/oie/pre-arrival-and-settling-in/settling-in-guide/your-first-weeks/graduate-checklist.html |
| 3 | cmu | cmu housing info | https://www.cmu.edu/oie/pre-arrival-and-settling-in/settling-in-guide/housing.html|
| 4 | cmu.guide | dining options | https://cmu.guide/meal-plans/ |
| 5 | cmu | student handbook  | pdf document |
| 6 | reddit | pros and cons of cmu | https://www.reddit.com/r/cmu/comments/1s54fio/help_me_fall_in_love_with_cmu/|
| 7 | reddit | study spots | https://www.reddit.com/r/cmu/comments/1j07hl1/best_spots_on_campus/ |
| 8 |reddit | housing | https://www.reddit.com/r/cmu/comments/16y9hn/reference_thread_everything_you_ever_wanted_to/|
| 9 | reddit | off-campus and on-campus activities | https://www.reddit.com/r/cmu/comments/1db47r/reference_thread_everything_you_ever_wanted_to/  |
| 10 | Tepper  | Tepper Survival guide | pdf document |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:** 
Dynamic, roughly 400-800 characters per semantic block

**Overlap:**
Adaptive, sentences that cross semantic break thresholds are carried over to preserve continuity

**Reasoning:**
Since some of the documents are multi-page guides standard fixed chunking might not be suitable. Slicing text exactly at a character limit could split sentences and meaning leading to chunks without meaning.

So with semantic chunking, the system will parse the documents sentence by sentence, compute embeddings, and only insert a chunk boundary when the semantic distance between sentences crosses a calculated threshold.
---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:**
all-MiniLM-L6-v2 via sentence-transformers

**Top-k:**
k = 5

**Production tradeoff reflection:**
If deploying this for a real university-wide product without budget constraints, all-MiniLM-L6-v2 would likely be replaced due to its restrictive 256-token context window. 
Domain-Specific Jargon: CMU students rely heavily on lingo (e.g., CUC, Andrew IDs, CUC). A baseline model might misinterpret these. In production, we would consider implementing a robust Hybrid Search to guarantee exact keyword matches for specific location names like "CUC".
---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | Which specific bus lines should a student take to get from the Oakland campus to grocery stores in Shadyside? | 75 or 71B |
| 2 | What can students do for free with their andrew ID | Museums, art pass, fress buses. Also should include subscriptions.|
| 3 | What do students say are good study spots around campus | Hunt, Sorells libraries|
| 4 | What do students think are good on-campus and off-campus dining spots? | Correct distinction between on-campus and off-campus spots. Example, Scottys, Stackd, Millies - on-campus. Noodlehead, Shah's Halal place, Senyai - off-campus |
| 5 | What housing options do first year graduate students have? | Shadyside, Okland, Squirrel Hull. No dorms available for graduate students.|

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1.
Some documents have images, this could lead to issues in chunking
2.
Sources include multi-page docs and reddit threads so there could be issues in boundaries or tone that might not translate well into embeddings

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

graph TD
    A[Document Ingestion: pdfplumber & Raw Text Files] --> B[Chunking Strategy: Semantic Chunking via Sentence-Transformers]
    B --> C[Embedding & Vector Store: all-MiniLM-L6-v2 + ChromaDB]
    C --> D[Retrieval Approach: top-k = 5 Semantic Search]
    E[User Query via Gradio Interface] --> D
    D --> F[Grounded Generation: Llama-3.3-70b via Groq API]
    F --> G[Cited Answer + Sources]
---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**
**Tool:** Claude 3.5 Sonnet / ChatGPT
**Input:** My "Domain", "Documents", and "Chunking Strategy" sections
**Expected Output:** A Python script that reads local text/PDF files, splits sentences, calculates embedding distance thresholds, and groups sentences into 400-800 character semantic chunks.
**Verification:** I will print out 5 random chunks to confirm they are self-contained thoughts and assert that no chunks are empty or filled with raw HTML/formatting artifacts.

**Milestone 4 — Embedding and retrieval:**
**Tool:** Claude 3.5 Sonnet / GitHub Copilot
**Input:** My "Retrieval Approach" section and the architecture diagram.
**Expected Output:** A script that initializes a local ChromaDB instance, generates embeddings using `all-MiniLM-L6-v2`, stores the chunks along with metadata, and exposes a `retrieve(query, k=5)` function.
**Verification:** I will query the vector store using 3 of my test questions and verify that the distance scores are under 0.5 and the returned text directly answers the query.

**Milestone 5 — Generation and interface:**
**Tool:** Claude 3.5 Sonnet
**Input:** My "Evaluation Plan" and the CodePath boilerplate code for the Gradio interface.
**Expected Output:** A script that hooks the Groq client (`llama-3.3-70b-versatile`) into the retrieval pipeline. The system prompt must strictly enforce that answers are derived only from the retrieved context and programmatically append the sources.
 **Verification:** I will test it with an out-of-scope question (e.g., "What is the weather in Paris?") to confirm it safely declines to answer rather than hallucinating.