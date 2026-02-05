"""Quick test to see actual answers for a few questions."""
import asyncio
import sys
sys.path.insert(0, ".")

from app.services.rag.service import get_rag_service

TEST_QUESTIONS = [
    "What are the key benefits of LIC's Jeevan Umang?",
    "Who is eligible for LIC's Amritbaal Plan?",
    "What is the death benefit under Jeevan Utsav?",
]

async def main():
    service = get_rag_service()

    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'='*80}")
        print(f"Q{i}: {q}")
        print("="*80)

        result = await service.answer_query(
            query=q,
            use_hybrid_search=True,
            use_auto_filters=True,
            limit=5,
        )

        print(f"\n📝 ANSWER:\n{result['answer']}")
        print(f"\n📚 SOURCES:")
        for c in result.get("citations", [])[:3]:
            print(f"  - {c.get('source', 'unknown')} (score: {c.get('score', 0):.3f})")

if __name__ == "__main__":
    asyncio.run(main())
