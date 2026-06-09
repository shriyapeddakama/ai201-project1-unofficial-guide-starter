"""Milestone 6: run the full system on all 5 evaluation questions.

For each planning.md eval query this prints the grounded answer, the source
documents cited, and the best retrieval distance -- the raw material for the
Evaluation Report table in the README.

Run from the project root (needs GROQ_API_KEY in .env):
    python -m src.evaluate
"""

from src.config import EVAL_QUERIES
from src.generate import ask


def main() -> None:
    for i, query in enumerate(EVAL_QUERIES, 1):
        out = ask(query)
        best = out["results"][0]["distance"] if out["results"] else float("nan")
        print("=" * 90)
        print(f"Q{i}: {query}")
        print(f"(best retrieval distance: {best:.3f})")
        print("-" * 90)
        print(out["answer"])
        print("\nRetrieved from:")
        for s in (out["sources"] or ["(none - declined)"]):
            print(f"  - {s}")
        print()


if __name__ == "__main__":
    main()
