"""
Test LLM classifier on the ACTUAL misclassified chunks from Qdrant.

These are real examples where regex returns "general" but the content
is clearly eligibility-related (tables with age column headers).
"""
import sys
sys.path.insert(0, ".")

from app.services.ingestion.smart_chunking.llm_section_classifier import (
    LLMSectionClassifier,
    classify_with_fallback,
)
from app.services.ingestion.chunkers import classify_section


# These are ACTUAL chunks from Qdrant that were tagged "general" but contain eligibility content
# (based on our earlier debug_chunks.py analysis)
HARD_CASES = [
    # Case 1: Real table from LIC_Jeevan Utsav.pdf page 2
    # Note: No "eligibility" keyword, just column headers with age info
    {
        "text": """| | |
| --- | --- | --- | --- | --- |
| | Minimum | Maximum |
| At entry of the Life Assured | 90 days (completed) | 50 years (nearer birthday) |
| Policy Term | 15 years | 25 years |
| Premium Paying Term | 5 years, 6 years, 7 years, 8 years and 12 years | |
| Sum Assured on Maturity | `1,00,000 | No limit |""",
        "expected": "eligibility",
        "description": "Jeevan Utsav eligibility table (real)"
    },

    # Case 2: Real table from LIC_Jeevan Akshay-VII
    {
        "text": """| | | Minimum | Maximum |
| --- | --- | --- | --- |
| Purchase Price | ` 1,50,000 | No limit |
| At Entry | Single Annuitant | 30 years (completed) | 85 years (nearer birthday) |
| | Primary Annuitant (in case of Joint Life Annuity) | 30 years (completed) | 85 years (nearer birthday) |
| | Secondary Annuitant (in case of Joint Life Annuity) | 30 years (completed) | 85 years (nearer birthday) |""",
        "expected": "eligibility",
        "description": "Jeevan Akshay-VII entry age table (real)"
    },

    # Case 3: Real from Amritbaal page 21
    {
        "text": """| | Minimum | Maximum |
| --- | --- | --- |
| At entry of the Life Assured | 0 (91 days completed) | 13 years (nearer birthday) |
| At entry of the Proposer | 18 years (completed) | 57 years (nearer birthday) |
| Policy Term | 20 – Age at entry | 25 years |
| Premium Paying Term | (Policy term – 3 years), subject to | |
| Sum Assured on Maturity | `1,00,000 | No limit |""",
        "expected": "eligibility",
        "description": "Amritbaal entry criteria table (real)"
    },

    # Case 4: Pure text about plan overview (should stay general)
    {
        "text": """LIC's New Jeevan Anand is a participating non-linked plan which offers an attractive combination of protection and savings. This combination provides financial protection against death throughout the lifetime of the policyholder with the payment of lumpsum at the end of selected policy term in case of his survival.""",
        "expected": "general",
        "description": "General plan overview"
    },
]


def main():
    print("=" * 70)
    print("Testing LLM Classifier on HARD CASES (actual misclassified chunks)")
    print("=" * 70)

    classifier = LLMSectionClassifier(use_small_model=True)

    correct_regex = 0
    correct_llm = 0

    for i, case in enumerate(HARD_CASES, 1):
        text = case["text"]
        expected = case["expected"]
        desc = case["description"]

        print(f"\n[Test {i}] {desc}")
        print("-" * 50)

        # Test regex classification
        regex_result = classify_section(text)
        regex_ok = regex_result == expected
        if regex_ok:
            correct_regex += 1

        # Test LLM classification
        llm_result = classifier.classify(text)
        llm_ok = llm_result.section_type == expected
        if llm_ok:
            correct_llm += 1

        # Test fallback
        fallback_result = classify_with_fallback(text, regex_result, classifier)

        print(f"Expected: {expected}")
        print(f"Regex:    {regex_result:12} {'✅' if regex_ok else '❌'}")
        print(f"LLM:      {llm_result.section_type:12} {'✅' if llm_ok else '❌'} (conf: {llm_result.confidence:.2f})")
        print(f"Fallback: {fallback_result:12}")
        print(f"Reasoning: {llm_result.reasoning}")

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"Regex accuracy:  {correct_regex}/{len(HARD_CASES)} ({100*correct_regex/len(HARD_CASES):.0f}%)")
    print(f"LLM accuracy:    {correct_llm}/{len(HARD_CASES)} ({100*correct_llm/len(HARD_CASES):.0f}%)")

    improvement = correct_llm - correct_regex
    if improvement > 0:
        print(f"\n🎉 LLM improves classification by {improvement} cases!")
    elif improvement < 0:
        print(f"\n⚠️  LLM performs worse by {-improvement} cases")
    else:
        print(f"\n📊 Both methods perform equally")


if __name__ == "__main__":
    main()
