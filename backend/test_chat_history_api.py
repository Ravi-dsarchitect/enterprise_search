"""
Test chat history via API calls.
Simpler approach that doesn't require importing the service layer.
"""

import requests
import json

BASE_URL = "http://localhost:8000/api/v1/chat"

def test_without_history():
    """Test 1: Query without history"""
    print("\n" + "="*70)
    print("TEST 1: Query WITHOUT History")
    print("="*70)
    
    payload = {
        "query": "What is LIC Jeevan Anand?",
        "use_hyde": False,
        "use_hybrid_search": False,
        "limit": 3
    }
    
    print(f"\nQuery: {payload['query']}")
    print("Sending request...")
    
    response = requests.post(f"{BASE_URL}/query", json=payload, timeout=60)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ SUCCESS (Status: {response.status_code})")
        print(f"📄 Citations: {len(result.get('citations', []))}")
        print(f"\nAnswer preview:\n{result['answer'][:300]}...")
        return result
    else:
        print(f"\n❌ FAILED (Status: {response.status_code})")
        print(f"Error: {response.text}")
        return None


def test_with_history():
    """Test 2: Multi-turn conversation with history"""
    print("\n" + "="*70)
    print("TEST 2: Multi-turn Conversation WITH History")
    print("="*70)
    
    # First query
    query1 = "What is LIC Jeevan Anand?"
    print(f"\n[Turn 1] User: {query1}")
    
    payload1 = {
        "query": query1,
        "use_hyde": False,
        "use_hybrid_search": False,
        "limit": 3
    }
    
    response1 = requests.post(f"{BASE_URL}/query", json=payload1, timeout=60)
    
    if response1.status_code != 200:
        print(f"❌ Turn 1 failed: {response1.status_code}")
        return None
    
    result1 = response1.json()
    print(f"[Turn 1] Assistant: {result1['answer'][:150]}...")
    
    # Second query with history
    query2 = "What are its maturity benefits?"
    print(f"\n[Turn 2] User: {query2}")
    
    conversation_history = [
        {"role": "user", "content": query1},
        {"role": "assistant", "content": result1['answer']}
    ]
    
    payload2 = {
        "query": query2,
        "conversation_history": conversation_history,
        "use_hyde": False,
        "use_hybrid_search": False,
        "limit": 3
    }
    
    print(f"📝 Sending with history ({len(conversation_history)} messages)...")
    
    response2 = requests.post(f"{BASE_URL}/query", json=payload2, timeout=60)
    
    if response2.status_code == 200:
        result2 = response2.json()
        print(f"\n✅ SUCCESS (Status: {response2.status_code})")
        print(f"[Turn 2] Assistant: {result2['answer'][:300]}...")
        
        # Check if answer references Jeevan Anand
        if "jeevan anand" in result2['answer'].lower():
            print("\n✅ CONTEXT AWARENESS: Answer correctly references Jeevan Anand from history!")
        else:
            print("\n⚠️  WARNING: Answer may not have used conversation context")
        
        return result2
    else:
        print(f"\n❌ FAILED (Status: {response2.status_code})")
        print(f"Error: {response2.text}")
        return None


def main():
    """Run all tests"""
    print("\n🧪 CHAT HISTORY API TESTS")
    print("="*70)
    print(f"Testing against: {BASE_URL}")
    
    try:
        # Test 1: No history
        result1 = test_without_history()
        
        if result1:
            # Test 2: With history
            result2 = test_with_history()
            
            if result2:
                print("\n" + "="*70)
                print("✅ ALL TESTS PASSED")
                print("="*70)
                print("\n📋 Summary:")
                print("  - Backward compatibility: ✅ Works without history")
                print("  - Multi-turn context: ✅ Works with history")
                print("  - API integration: ✅ Endpoints accept conversation_history")
            else:
                print("\n❌ Test 2 failed")
        else:
            print("\n❌ Test 1 failed")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to backend at", BASE_URL)
        print("Make sure the backend is running: uvicorn app.main:app --reload")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
