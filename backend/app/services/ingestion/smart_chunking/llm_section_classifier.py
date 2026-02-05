"""
LLM-based section type classifier for chunks.

Uses a fast LLM to classify chunk content into section types based on semantic
understanding, not just keyword matching. This is much more accurate than regex
for edge cases like tables with headers "Minimum Age at Entry" (which is eligibility
content but doesn't contain the word "eligibility").

Usage modes:
1. Full LLM classification: Every chunk gets LLM classification (slower, most accurate)
2. Fallback mode: Only use LLM when regex returns "general" (balanced speed/accuracy)
3. Batch mode: Classify multiple chunks in one LLM call (efficient for large documents)
"""

import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from app.core.config import settings


@dataclass
class ClassificationResult:
    """Result of section classification."""
    section_type: str
    confidence: float
    reasoning: str


# Valid section types (must match what's used in the rest of the system)
VALID_SECTION_TYPES = [
    "eligibility",  # Age limits, entry requirements, who can buy
    "benefits",     # Death benefit, maturity benefit, survival benefit, bonus
    "premium",      # Premium payment, modes, rates, rebates
    "surrender",    # Surrender value, paid-up, lapse, revival, free look
    "loan",         # Policy loan, loan facility
    "tax",          # Tax benefits, 80C, 10(10D)
    "rider",        # Additional benefits, ADB, CI, WOP
    "claim",        # Claim process, settlement, documents required
    "annuity",      # Pension, vesting, annuity options
    "fund",         # NAV, ULIP, fund options
    "charges",      # Fees, deductions, mortality charges
    "exclusions",   # Not covered, limitations
    "general",      # Default/unknown
]


class LLMSectionClassifier:
    """
    Classifies chunk content into section types using LLM.

    The LLM understands context better than regex:
    - Table with "Minimum Age | Maximum Age" → eligibility
    - Table with "Sum Assured | Premium" → premium
    - Text about "death benefit calculation" → benefits
    """

    def __init__(self, use_small_model: bool = True):
        """
        Initialize the classifier.

        Args:
            use_small_model: Use the metadata model (smaller/faster) instead of main LLM
        """
        self._llm = None
        self.use_small_model = use_small_model

    def _get_llm(self):
        """Lazy-load the LLM."""
        if self._llm is None:
            provider = settings.LLM_PROVIDER.lower()
            model = settings.LLM_METADATA_MODEL if self.use_small_model else settings.LLM_MODEL_NAME

            if provider == "ollama":
                from langchain_ollama import ChatOllama
                self._llm = ChatOllama(
                    model=model,
                    temperature=0.0,  # Deterministic for classification
                    base_url=settings.OLLAMA_BASE_URL,
                )
            else:
                from app.services.rag.generator import LLMFactory
                generator = LLMFactory.create_generator()
                self._llm = generator.llm

            print(f"  [LLMSectionClassifier] Loaded model: {model}")
        return self._llm

    def classify(self, chunk_text: str, context: Optional[str] = None) -> ClassificationResult:
        """
        Classify a single chunk into a section type.

        Args:
            chunk_text: The text content of the chunk
            context: Optional context like document title or preceding heading

        Returns:
            ClassificationResult with section_type, confidence, and reasoning
        """
        system_prompt = self._get_system_prompt()

        user_content = f"Classify this content:\n\n{chunk_text[:2000]}"  # Limit to 2000 chars
        if context:
            user_content = f"Document context: {context}\n\n{user_content}"

        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        try:
            llm = self._get_llm()
            response = llm.invoke(messages)
            return self._parse_response(response.content)
        except Exception as e:
            print(f"  [LLMSectionClassifier] Error: {e}")
            return ClassificationResult(
                section_type="general",
                confidence=0.0,
                reasoning=f"Classification failed: {e}"
            )

    def classify_batch(
        self,
        chunks: List[Tuple[str, Optional[str]]],
        max_per_call: int = 10
    ) -> List[ClassificationResult]:
        """
        Classify multiple chunks efficiently in batches.

        Args:
            chunks: List of (chunk_text, optional_context) tuples
            max_per_call: Maximum chunks to classify in one LLM call

        Returns:
            List of ClassificationResults in same order as input
        """
        results = []

        for i in range(0, len(chunks), max_per_call):
            batch = chunks[i:i + max_per_call]
            batch_results = self._classify_batch_single(batch)
            results.extend(batch_results)

        return results

    def _classify_batch_single(
        self,
        chunks: List[Tuple[str, Optional[str]]]
    ) -> List[ClassificationResult]:
        """Classify a batch of chunks in a single LLM call."""
        system_prompt = self._get_batch_system_prompt()

        # Build numbered list of chunks
        chunk_list = []
        for i, (text, context) in enumerate(chunks, 1):
            preview = text[:500].replace('\n', ' ').strip()
            if context:
                chunk_list.append(f"[{i}] Context: {context}\nContent: {preview}")
            else:
                chunk_list.append(f"[{i}] {preview}")

        user_content = "Classify these chunks:\n\n" + "\n\n".join(chunk_list)

        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        try:
            llm = self._get_llm()
            response = llm.invoke(messages)
            return self._parse_batch_response(response.content, len(chunks))
        except Exception as e:
            print(f"  [LLMSectionClassifier] Batch error: {e}")
            return [
                ClassificationResult("general", 0.0, f"Batch classification failed: {e}")
                for _ in chunks
            ]

    def _get_system_prompt(self) -> str:
        """Get the system prompt for single-chunk classification."""
        return f"""You are a document section classifier for insurance documents.
Classify the given content into exactly ONE section type.

VALID SECTION TYPES:
- eligibility: Age limits, entry requirements, who can buy, age at entry tables, minimum/maximum age
- benefits: Death benefit, maturity benefit, survival benefit, bonus, guaranteed additions
- premium: Premium payment, premium rates, payment modes, rebates, premium tables
- surrender: Surrender value, paid-up, lapse, revival, free look period, GSV/SSV
- loan: Policy loan facility, loan interest, loan eligibility
- tax: Tax benefits, Section 80C, 10(10D), tax deductions
- rider: Additional riders, ADB, critical illness, waiver of premium
- claim: Claim process, settlement, documents required, nominee
- annuity: Pension, vesting, annuity options, commutation, deferment
- fund: NAV, ULIP funds, fund performance, fund switching
- charges: Fees, deductions, mortality charges, allocation charges
- exclusions: Not covered, limitations, suicide clause, waiting period
- general: Only if none of the above clearly apply

IMPORTANT RULES:
1. Tables with age columns (Minimum Age, Maximum Age, Entry Age) → "eligibility"
2. Tables with premium/sum assured columns → "premium"
3. Tables with benefit calculations → "benefits"
4. Content about "who can buy", age requirements → "eligibility"
5. Return ONLY valid JSON, no markdown.

Return JSON:
{{"section_type": "one of the types above", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

    def _get_batch_system_prompt(self) -> str:
        """Get the system prompt for batch classification."""
        return f"""You are a document section classifier for insurance documents.
Classify each numbered chunk into exactly ONE section type.

VALID SECTION TYPES:
- eligibility: Age limits, entry requirements, who can buy, age at entry tables
- benefits: Death benefit, maturity benefit, survival benefit, bonus
- premium: Premium payment, rates, payment modes, premium tables
- surrender: Surrender value, paid-up, lapse, revival, free look
- loan: Policy loan facility
- tax: Tax benefits, Section 80C, 10(10D)
- rider: Additional riders, ADB, critical illness, waiver
- claim: Claim process, settlement, documents required
- annuity: Pension, vesting, annuity options
- fund: NAV, ULIP funds, fund options
- charges: Fees, deductions, mortality charges
- exclusions: Not covered, limitations
- general: Only if none of the above clearly apply

IMPORTANT RULES:
1. Tables with age columns → "eligibility"
2. Tables with premium columns → "premium"
3. Tables with benefit calculations → "benefits"

Return JSON array (one object per chunk, same order):
[{{"chunk": 1, "section_type": "type", "confidence": 0.9, "reasoning": "brief"}}, ...]"""

    def _parse_response(self, content: str) -> ClassificationResult:
        """Parse LLM response for single classification."""
        try:
            # Clean markdown
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            data = json.loads(content)

            section_type = data.get("section_type", "general").lower()
            if section_type not in VALID_SECTION_TYPES:
                section_type = "general"

            return ClassificationResult(
                section_type=section_type,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", "")
            )
        except (json.JSONDecodeError, KeyError) as e:
            # Try to extract section type from plain text
            content_lower = content.lower()
            for st in VALID_SECTION_TYPES:
                if st in content_lower:
                    return ClassificationResult(st, 0.5, "Extracted from text")
            return ClassificationResult("general", 0.0, f"Parse failed: {e}")

    def _parse_batch_response(self, content: str, expected_count: int) -> List[ClassificationResult]:
        """Parse LLM response for batch classification."""
        try:
            # Clean markdown
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            data = json.loads(content)

            if not isinstance(data, list):
                data = [data]

            results = []
            for item in data:
                section_type = item.get("section_type", "general").lower()
                if section_type not in VALID_SECTION_TYPES:
                    section_type = "general"

                results.append(ClassificationResult(
                    section_type=section_type,
                    confidence=float(item.get("confidence", 0.5)),
                    reasoning=item.get("reasoning", "")
                ))

            # Pad with general if we got fewer results than expected
            while len(results) < expected_count:
                results.append(ClassificationResult("general", 0.0, "Missing from response"))

            return results[:expected_count]

        except Exception as e:
            return [
                ClassificationResult("general", 0.0, f"Batch parse failed: {e}")
                for _ in range(expected_count)
            ]


def classify_with_fallback(
    chunk_text: str,
    regex_result: str,
    classifier: Optional[LLMSectionClassifier] = None,
    context: Optional[str] = None
) -> str:
    """
    Classify using regex first, fall back to LLM if result is "general".

    This is the recommended approach for production:
    - Fast: Regex handles most clear-cut cases
    - Accurate: LLM handles ambiguous cases (like tables)

    Args:
        chunk_text: Text to classify
        regex_result: Result from classify_section() regex function
        classifier: LLMSectionClassifier instance (created if None)
        context: Optional document context

    Returns:
        Section type string
    """
    # If regex found a specific type, trust it
    if regex_result != "general":
        return regex_result

    # Regex returned "general" - check if this might be misclassified
    # Quick heuristic: if it looks like a table with age/premium data, use LLM
    might_be_misclassified = (
        "|" in chunk_text or  # Table format
        "age" in chunk_text.lower() or
        "minimum" in chunk_text.lower() or
        "maximum" in chunk_text.lower() or
        "entry" in chunk_text.lower()
    )

    if not might_be_misclassified:
        return "general"

    # Use LLM for potentially misclassified chunks
    if classifier is None:
        classifier = LLMSectionClassifier()

    result = classifier.classify(chunk_text, context)

    if result.confidence >= 0.6 and result.section_type != "general":
        print(f"  [LLM Reclassified] general → {result.section_type} (conf: {result.confidence:.2f})")
        return result.section_type

    return "general"
