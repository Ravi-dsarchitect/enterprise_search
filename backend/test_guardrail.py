"""
Test the QueryGuardrail classifier with example queries.

Requires Ollama running with the configured model (e.g. qwen2.5:3b).

Usage:
    cd /home/ec2-user/nakul/enterprise_search/backend
    python test_guardrail.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.rag.guardrail import QueryGuardrail

# Test cases: (query, expected_allowed)
TEST_CASES = [
    # --- Should be ALLOWED ---
    ("What are the benefits of Jeevan Umang?", True),
    ("What is the eligibility for LIC pension plans?", True),
    ("How to claim death benefit?", True),
    ("What riders are available with Bima Ratna?", True),
    ("Tax benefits under section 80C for LIC policies", True),
    ("What is the surrender value after 5 years?", True),
    ("Hi, I need help", True),
    ("Tell me more about that", True),
    ("What is the premium for a 30 year old?", True),
    ("Compare Jeevan Utsav and Jeevan Umang", True),

    # --- Should be REJECTED ---
    ("What is the capital of France?", False),
    ("Write me a Python script to sort a list", False),
    ("What is the best recipe for biryani?", False),
    ("Who won the cricket world cup in 2023?", False),
    ("Tell me a joke about cats", False),
    ("How do I install TensorFlow?", False),
]


def main():
    guardrail = QueryGuardrail()

    passed = 0
    failed = 0
    results = []

    print("=" * 70)
    print("QUERY GUARDRAIL TEST")
    print("=" * 70)

    for query, expected_allowed in TEST_CASES:
        start = time.time()
        result = guardrail.check(query)
        elapsed = time.time() - start

        status = "✅ PASS" if result.allowed == expected_allowed else "❌ FAIL"
        if result.allowed == expected_allowed:
            passed += 1
        else:
            failed += 1

        results.append((query, expected_allowed, result.allowed, result.reason, elapsed))
        print(f"\n{status} | {elapsed:.2f}s")
        print(f"  Query:    {query}")
        print(f"  Expected: {'ALLOW' if expected_allowed else 'REJECT'}")
        print(f"  Got:      {'ALLOW' if result.allowed else 'REJECT'}")
        print(f"  Reason:   {result.reason}")

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{len(TEST_CASES)} passed, {failed} failed")
    print("=" * 70)

    if failed > 0:
        print("\nFailed cases:")
        for query, expected, actual, reason, _ in results:
            if actual != expected:
                print(f"  - \"{query}\" → expected {'ALLOW' if expected else 'REJECT'}, got {'ALLOW' if actual else 'REJECT'} ({reason})")
        sys.exit(1)


if __name__ == "__main__":
    main()
