"""
Test LLM-based section classifier on content that regex misclassifies.

This tests the fix for the issue where eligibility tables with headers like
"Minimum Age at Entry | Maximum Age at Entry" were classified as "general"
because the word "eligibility" wasn't present.
"""
import sys
sys.path.insert(0, ".")

from app.services.ingestion.smart_chunking.llm_section_classifier import (
    LLMSectionClassifier,
    classify_with_fallback,
)
from app.services.ingestion.chunkers import classify_section

# Sample content that regex misclassifies as "general"
TEST_CASES = [
    # Case 1: Eligibility table from Jeevan Utsav
    {
        "text": """| Particulars | Minimum | Maximum |
| --- | --- | --- |
| Age at entry of Life Assured | 90 days (completed) | 50 years |
| Policy Term | 15 years | 25 years |
| Premium Paying Term | 5 years | 12 years |
| Sum Assured on Maturity | Rs. 1,00,000 | No limit |""",
        "expected": "eligibility",
        "description": "Eligibility table from Jeevan Utsav"
    },

    # Case 2: Age criteria table without "eligibility" keyword
    {
        "text": """Minimum Age at Entry | Maximum Age at Entry | Minimum Policy Term | Maximum Policy Term
18 years | 65 years | 10 years | 40 years
The minimum age at maturity shall be 70 years.""",
        "expected": "eligibility",
        "description": "Age entry table"
    },

    # Case 3: Premium table (should NOT be eligibility)
    {
        "text": """| Sum Assured | Annual Premium | Half-Yearly | Quarterly | Monthly |
| --- | --- | --- | --- | --- |
| Rs. 1,00,000 | Rs. 8,500 | Rs. 4,350 | Rs. 2,200 | Rs. 750 |
| Rs. 2,00,000 | Rs. 17,000 | Rs. 8,700 | Rs. 4,400 | Rs. 1,500 |""",
        "expected": "premium",
        "description": "Premium rates table"
    },

    # Case 4: Benefits description
    {
        "text": """Death Benefit: In case of unfortunate death of the Life Assured during the policy term,
the following amount shall be payable to the nominee:
- 10 times the Annualized Premium, or
- Sum Assured on Death, or
- 105% of premiums paid""",
        "expected": "benefits",
        "description": "Death benefit description"
    },

    # Case 5: Truly general content
    {
        "text": """LIC of India is one of the largest insurance companies in the world.
It was established in 1956 and has been serving millions of policyholders.""",
        "expected": "general",
        "description": "General company information"
    },
]


def main():
    print("=" * 70)
    print("Testing LLM Section Classifier")
    print("=" * 70)

    classifier = LLMSectionClassifier(use_small_model=True)

    results = []

    for i, case in enumerate(TEST_CASES, 1):
        text = case["text"]
        expected = case["expected"]
        desc = case["description"]

        print(f"\n[Test {i}] {desc}")
        print(f"Expected: {expected}")

        # Test regex classification
        regex_result = classify_section(text)
        print(f"Regex result: {regex_result}")

        # Test LLM classification
        llm_result = classifier.classify(text)
        print(f"LLM result: {llm_result.section_type} (confidence: {llm_result.confidence:.2f})")
        print(f"LLM reasoning: {llm_result.reasoning}")

        # Test fallback function
        fallback_result = classify_with_fallback(text, regex_result, classifier)
        print(f"Fallback result: {fallback_result}")

        # Check if correct
        is_correct = llm_result.section_type == expected
        results.append({
            "case": i,
            "description": desc,
            "expected": expected,
            "regex": regex_result,
            "llm": llm_result.section_type,
            "fallback": fallback_result,
            "correct": is_correct
        })

        print(f"✅ PASS" if is_correct else f"❌ FAIL (expected {expected})")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    correct = sum(1 for r in results if r["correct"])
    print(f"LLM Accuracy: {correct}/{len(results)} ({100*correct/len(results):.0f}%)")

    print("\nComparison: Regex vs LLM")
    print("-" * 40)
    for r in results:
        regex_ok = "✅" if r["regex"] == r["expected"] else "❌"
        llm_ok = "✅" if r["llm"] == r["expected"] else "❌"
        print(f"Case {r['case']}: Regex {regex_ok} ({r['regex']:12}) | LLM {llm_ok} ({r['llm']:12})")


if __name__ == "__main__":
    main()
