"""
Test script for chat history functionality.
Tests both backward compatibility (no history) and multi-turn conversations.
"""

import asyncio
import json
import sys
from typing import List, Dict

# Add parent to path for imports
sys.path.insert(0, ".")

from app.services.rag.service import RAGService


async def test_without_history():
    """Test 1: Query without history (backward compatibility)"""
    print("\n" + "="*70)
    print("TEST 1: Query WITHOUT History (Backward Compatibility)")
    print("="*70)
    
    service = RAGService()
    
    query = "What is LIC Jeevan Anand?"
    print(f"\nQuery: {query}")
    
    result = await service.answer_query(
        query=query,
        use_hyde=False,
        use_hybrid_search=False,
        limit=3
    )
    
    print(f"\n✅ Answer received ({len(result['answer'])} chars)")
    print(f"📄 Citations: {len(result['citations'])}")
    print(f"\nAnswer preview:\n{result['answer'][:300]}...")
    
    return result


async def test_with_history():
    """Test 2: Multi-turn conversation with history"""
    print("\n" + "="*70)
    print("TEST 2: Multi-turn Conversation WITH History")
    print("="*70)
    
    service = RAGService()
    conversation_history = []
    
    # First query
    query1 = "What is LIC Jeevan Anand?"
    print(f"\n[Turn 1] User: {query1}")
    
    result1 = await service.answer_query(
        query=query1,
        conversation_history=None,
        use_hyde=False,
        use_hybrid_search=False,
        limit=3
    )
    
    print(f"[Turn 1] Assistant: {result1['answer'][:150]}...")
    
    # Add to history
    conversation_history.append({"role": "user", "content": query1})
    conversation_history.append({"role": "assistant", "content": result1['answer']})
    
    # Second query (context-dependent)
    query2 = "What are its maturity benefits?"
    print(f"\n[Turn 2] User: {query2}")
    print(f"📝 History: {len(conversation_history)} messages")
    
    result2 = await service.answer_query(
        query=query2,
        conversation_history=conversation_history,
        use_hyde=False,
        use_hybrid_search=False,
        limit=3
    )
    
    print(f"[Turn 2] Assistant: {result2['answer'][:300]}...")
    
    # Check if answer references Jeevan Anand
    if "jeevan anand" in result2['answer'].lower():
        print("\n✅ SUCCESS: Answer correctly references Jeevan Anand from history!")
    else:
        print("\n⚠️  WARNING: Answer may not have used conversation context")
    
    return result2


async def test_streaming_with_history():
    """Test 3: Streaming with history"""
    print("\n" + "="*70)
    print("TEST 3: Streaming Query WITH History")
    print("="*70)
    
    service = RAGService()
    
    # Prepare history
    conversation_history = [
        {"role": "user", "content": "What is LIC Jeevan Anand?"},
        {"role": "assistant", "content": "LIC Jeevan Anand is a comprehensive life insurance plan..."}
    ]
    
    query = "What are the premium payment options?"
    print(f"\nQuery: {query}")
    print(f"📝 History: {len(conversation_history)} messages")
    print("\nStreaming response:")
    print("-" * 70)
    
    full_answer = ""
    async for event in service.answer_query_stream(
        query=query,
        conversation_history=conversation_history,
        use_hyde=False,
        use_hybrid_search=False,
        limit=3
    ):
        if event["type"] == "token":
            token = event["data"]
            print(token, end="", flush=True)
            full_answer += token
        elif event["type"] == "done":
            print("\n" + "-" * 70)
            print(f"\n✅ Streaming complete ({len(full_answer)} chars)")
    
    return full_answer


async def main():
    """Run all tests"""
    print("\n🧪 CHAT HISTORY FUNCTIONALITY TESTS")
    print("="*70)
    
    try:
        # Test 1: No history
        await test_without_history()
        
        # Test 2: With history
        await test_with_history()
        
        # Test 3: Streaming with history
        await test_streaming_with_history()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
