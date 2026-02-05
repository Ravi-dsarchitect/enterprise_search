"""
Ask questions against the RAG pipeline.

Uses the best evaluated config:
  - Hybrid search (Vector + BM25): Hit@5=95.2%, source_accuracy=97.6%
  - No HyDE, no decomposition
  - Auto metadata filters enabled

Usage:
  python ask.py "What are the benefits of Jeevan Utsav?"
  python ask.py  # interactive mode
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.rag.service import get_rag_service


# Best config from evaluation results (2026-02-04)
BEST_CONFIG = {
    "use_hybrid_search": True,
    "use_hyde": False,
    "use_decomposition": False,
    "use_auto_filters": True,
    "limit": 5,
}


async def ask(question: str, verbose: bool = False) -> dict:
    """
    Ask a question and get an answer with citations.

    Args:
        question: The question to ask.
        verbose: If True, print retrieved chunks details.

    Returns:
        Dict with keys: query, answer, citations, generated_queries
    """
    service = get_rag_service()

    result = await service.answer_query(
        query=question,
        **BEST_CONFIG,
    )

    print(f"\nQ: {question}")
    print(f"\nA: {result['answer']}")

    if result.get("citations"):
        print(f"\n--- Sources ({len(result['citations'])}) ---")
        for i, c in enumerate(result["citations"], 1):
            source = c.get("source", "")
            score = c.get("score", 0)
            print(f"  [{i}] {source} (score={score:.4f})")

            if verbose:
                text_preview = c.get("text", "")[:200].replace("\n", " ")
                print(f"      {text_preview}...")

    return result


async def interactive():
    """Interactive question loop."""
    print("RAG Pipeline - Interactive Mode")
    print(f"Config: Hybrid={BEST_CONFIG['use_hybrid_search']}, "
          f"AutoFilters={BEST_CONFIG['use_auto_filters']}, "
          f"Limit={BEST_CONFIG['limit']}")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            break

        await ask(question, verbose=True)
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        asyncio.run(ask(question, verbose=True))
    else:
        asyncio.run(interactive())
