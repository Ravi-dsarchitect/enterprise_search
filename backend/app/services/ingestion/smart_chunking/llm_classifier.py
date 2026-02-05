"""
LLMContentClassifier - Optional LLM-based document classification.

Used when heuristic confidence is low. Disabled by default.
"""

import json
from typing import Optional

from app.services.ingestion.interfaces import ParsedDocument
from app.services.ingestion.smart_chunking.models import (
    ContentAnalysis,
    ContentStructure,
)
from app.core.config import settings


class LLMContentClassifier:
    """
    Uses LLM to classify document content type when heuristics are uncertain.

    Only used when:
    1. use_llm_classifier=True in SmartChunker
    2. Heuristic confidence < threshold
    """

    PROMPT = """Analyze this document excerpt and classify its structure type.

Document info:
- Filename: {filename}
- Pages: {page_count}
- Headings detected: {heading_count}
- Tables detected: {table_count}
- Initial analysis: {initial_structure} (confidence: {confidence:.0%})

First 2000 characters:
{content_preview}

Headings found:
{headings}

Classify the PRIMARY structure as one of:
- hierarchical: Nested sections and subsections (reports, manuals, policies)
- tabular: Primarily data tables
- qa: Question-and-answer format (FAQs)
- narrative: Long-form prose without strong structure
- slide: Presentation slides
- mixed: Combination

Respond with ONLY JSON:
{{
    "structure_type": "hierarchical|tabular|qa|narrative|slide|mixed",
    "confidence": 0.0-1.0,
    "recommended_chunker": "hierarchical|table_aware|qa|semantic|structured",
    "reasoning": "Brief explanation"
}}"""

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        """Lazy-load LLM."""
        if self._llm is None:
            provider = settings.LLM_PROVIDER.lower()
            model = settings.LLM_METADATA_MODEL or settings.LLM_MODEL_NAME

            if provider == "ollama":
                from langchain_ollama import ChatOllama

                self._llm = ChatOllama(
                    model=model,
                    temperature=0.1,
                    base_url=settings.OLLAMA_BASE_URL,
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}. Use 'ollama'.")

        return self._llm

    def classify(
        self,
        doc: ParsedDocument,
        initial_analysis: ContentAnalysis,
    ) -> ContentAnalysis:
        """Use LLM to classify document and refine analysis."""
        headings_str = "\n".join(f"- {h}" for h in doc.headings[:20]) or "(none)"

        prompt = self.PROMPT.format(
            filename=doc.filename,
            page_count=doc.total_pages,
            heading_count=initial_analysis.metrics.heading_count,
            table_count=initial_analysis.metrics.table_count,
            initial_structure=initial_analysis.primary_structure.value,
            confidence=initial_analysis.confidence,
            content_preview=doc.full_text[:2000],
            headings=headings_str,
        )

        from langchain_core.messages import HumanMessage

        try:
            llm = self._get_llm()
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()

            # Parse JSON
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            result = json.loads(content)

            # Map structure type
            structure_map = {
                "hierarchical": ContentStructure.HIERARCHICAL,
                "tabular": ContentStructure.TABULAR,
                "qa": ContentStructure.QA,
                "narrative": ContentStructure.NARRATIVE,
                "slide": ContentStructure.SLIDE,
                "mixed": ContentStructure.MIXED,
            }

            new_structure = structure_map.get(
                result.get("structure_type", "").lower(),
                initial_analysis.primary_structure,
            )

            # Update analysis
            return ContentAnalysis(
                primary_structure=new_structure,
                secondary_structure=initial_analysis.secondary_structure,
                metrics=initial_analysis.metrics,
                recommended_chunker=result.get(
                    "recommended_chunker", initial_analysis.recommended_chunker
                ),
                confidence=result.get("confidence", 0.8),
                supports_hierarchy=initial_analysis.supports_hierarchy,
                hierarchy_depth=initial_analysis.hierarchy_depth,
                major_sections=initial_analysis.major_sections,
                analysis_notes=initial_analysis.analysis_notes
                + [f"LLM: {result.get('reasoning', 'N/A')}"],
            )

        except Exception as e:
            print(f"[LLMContentClassifier] Error: {e}")
            return initial_analysis
