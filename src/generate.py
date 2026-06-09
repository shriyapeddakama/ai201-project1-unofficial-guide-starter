"""Milestone 5: grounded generation.

ask(question) ties the whole pipeline together: retrieve the top-k chunks,
build a context-only prompt, call Groq's llama-3.3-70b-versatile, and return
the answer together with the source documents it was allowed to draw from.

Grounding is enforced two ways:
  1. The system prompt forbids using any knowledge outside the provided
     context and mandates an exact refusal string when the context is
     insufficient.
  2. Source attribution is built in code from the retrieved chunks, not left
     to the model to add -- so citations can't be hallucinated. When the model
     refuses (no answer found in context), we return no sources.

Run from the project root (needs GROQ_API_KEY in .env):
    python -m src.generate
"""

import os

from dotenv import load_dotenv
from groq import Groq

from src.config import TOP_K, EVAL_QUERIES
from src.retrieve import retrieve

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
REFUSAL = "I don't have enough information on that."

SYSTEM_PROMPT = (
    "You are the CMU Unofficial Guide, a question-answering assistant for "
    "Carnegie Mellon students. You must answer using ONLY the information in "
    "the numbered context documents provided in the user's message. Follow "
    "these rules strictly:\n"
    "1. Do NOT use any outside or prior knowledge. If a fact is not stated in "
    "the context, you do not know it.\n"
    f"2. If the context does not contain enough information to answer the "
    f"question, reply with exactly this sentence and nothing else: \"{REFUSAL}\"\n"
    "3. Do not invent names, numbers, prices, bus routes, or places that are "
    "not present in the context.\n"
    "4. When you do answer, cite the supporting context items inline using "
    "their bracketed numbers, e.g. [1] or [2]."
)

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_context(results: list[dict]) -> str:
    """Render retrieved chunks as a numbered, source-labelled context block."""
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(f"[{i}] (source: {r['source']})\n{r['text']}")
    return "\n\n".join(blocks)


def ask(question: str, k: int = TOP_K) -> dict:
    """Answer `question` grounded in the retrieved context.

    Returns {"answer": str, "sources": list[str], "results": list[dict]}.
    `sources` lists the unique retrieved source documents in rank order, and is
    empty when the model declines to answer.
    """
    results = retrieve(question, k=k)
    context = build_context(results)

    user_msg = (
        f"Context documents:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )

    completion = _client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,  # deterministic, factual
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    answer = completion.choices[0].message.content.strip()

    # Programmatic source attribution: unique sources in rank order. If the
    # model refused, there is nothing to attribute.
    if REFUSAL.lower() in answer.lower():
        sources = []
    else:
        seen, sources = set(), []
        for r in results:
            if r["source"] not in seen:
                seen.add(r["source"])
                sources.append(r["source"])

    return {"answer": answer, "sources": sources, "results": results}


def main() -> None:
    # 2 in-scope eval queries + 1 out-of-scope query the docs can't answer.
    test_queries = [EVAL_QUERIES[2], EVAL_QUERIES[4],
                    "What is the weather in Paris today?"]
    for q in test_queries:
        print("=" * 90)
        print(f"Q: {q}")
        print("-" * 90)
        out = ask(q)
        print(out["answer"])
        print("\nRetrieved from:")
        if out["sources"]:
            for s in out["sources"]:
                print(f"  - {s}")
        else:
            print("  (none - model declined to answer)")
        print()


if __name__ == "__main__":
    main()
