from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import json
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.rag.generator import LLMFactory

class QueryAnalyzer:
    """
    Analyzes natural language queries to automatically extract metadata filter hints.
    Makes the search interface more intelligent and user-friendly.
    """
    
    def __init__(self):
        self.llm_generator = LLMFactory.create_generator()
    
    def extract_filters(self, query: str, confidence_threshold: float = 0.6) -> Optional[Dict[str, Any]]:
        """
        Extract metadata filters from natural language query using LLM.
        
        Args:
            query: Natural language search query
            confidence_threshold: Minimum confidence to apply filters (0-1)
        
        Returns:
            Dictionary of metadata filters or None if confidence too low
        """
        # Get current date for temporal calculations
        current_date = datetime.now().strftime("%Y-%m-%d")
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        system_prompt = f"""You are a query analysis expert for LIC insurance documents. Extract metadata filter hints from user queries.

=== DOCUMENT-LEVEL FIELDS ===
- category: [Policy, Circular, SOP, Manual, Report, Sales Brochure, Claim Guide, FAQ, Other]
- source: document filename (e.g., "LIC_Jeevan Umang.pdf")
- document_date: YYYY-MM-DD format
- plan_name: e.g. "Jeevan Umang", "Bima Ratna", "Jeevan Utsav"
- plan_number: e.g. "871", "943"
- plan_type: [Endowment, Whole Life, Money Back, Term Insurance, Pension/Annuity, ULIP, Child Plan]
- premium_type: [Single Premium, Limited Pay, Regular Premium]

=== CHUNK-LEVEL FIELDS (IMPORTANT for content-specific queries) ===
- section_type: [eligibility, benefits, financial, riders, tax, claims, general]
- content_type: [text, table]
- chunk_tags: Array of content tags like ["Death Benefit", "Tax Benefit", "Loan Facility", "Eligibility Criteria"]
- contains_age_info: true/false (has age limits, policy terms)
- contains_currency: true/false (has monetary amounts)

=== SECTION TYPE MAPPING (use for content-specific queries) ===
Query contains → section_type filter:
- "eligibility", "age", "who can buy", "entry age", "sum assured range", "policy term" → "eligibility"
- "benefits", "death benefit", "maturity benefit", "survival benefit", "bonus", "payout", "what do I get" → "benefits"
- "premium", "surrender value", "loan", "paid-up", "GSV", "SSV", "how much to pay" → "financial"
- "rider", "accidental death", "critical illness", "waiver of premium", "ADB", "WOP" → "riders"
- "tax", "80C", "10(10D)", "deduction", "tax benefit" → "tax"
- "claim", "settlement", "nominee", "documents required", "how to claim" → "claims"

=== CONTENT TYPE MAPPING ===
- "table", "comparison", "rates", "charges", "factor" → content_type: "table"

=== CHUNK TAGS (use when specific content is requested) ===
Common tags: Death Benefit, Maturity Benefit, Survival Benefit, Tax Benefit, Loan Facility,
Surrender, Grace Period, Premium Waiver, Accident Benefit, Critical Illness, Annuity Options,
Pension Plan, ULIP, Fund Options, Claim Process, Eligibility Criteria, etc.

=== PLAN TYPE MAPPING ===
- "pension", "retirement", "annuity" → plan_type: "Pension/Annuity"
- "child education", "child plan" → plan_type: "Child Plan"
- "term insurance", "pure protection" → plan_type: "Term Insurance"
- "ulip", "market linked" → plan_type: "ULIP"
- "money back", "survival benefit" → plan_type: "Money Back"
- "whole life", "lifelong" → plan_type: "Whole Life"
- "endowment", "savings" → plan_type: "Endowment"

=== TEMPORAL HINTS ===
- "recent", "latest" → document_date >= {thirty_days_ago}
- "this year" → document_date >= 2026-01-01

=== EXAMPLES ===
Q: "What is the eligibility for Jeevan Umang?"
→ plan_name: "Jeevan Umang", section_type: "eligibility"

Q: "Show me death benefit calculation"
→ section_type: "benefits", chunk_tags: ["Death Benefit"]

Q: "Tax benefits of pension plans"
→ plan_type: "Pension/Annuity", section_type: "tax"

Q: "Premium rates table for Bima Ratna"
→ plan_name: "Bima Ratna", content_type: "table", section_type: "financial"

Return JSON:
{{
  "filters": {{
    "category": "string or null",
    "source": "string or null",
    "document_date": {{"gte": "date", "lte": "date"}} or null,
    "plan_name": "string or null",
    "plan_number": "string or null",
    "plan_type": "string or null",
    "premium_type": "string or null",
    "section_type": "string or null",
    "content_type": "string or null",
    "chunk_tags": ["array"] or null,
    "contains_age_info": true/false or null,
    "contains_currency": true/false or null
  }},
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}

IMPORTANT: For content-specific queries, ALWAYS include section_type filter.
Return confidence < 0.5 only if query is very generic with no clear hints.
Return ONLY valid JSON, no markdown."""

        user_prompt = f"""Current date: {current_date}

Query: "{query}"

Extract filter hints:"""

        llm = self.llm_generator.llm
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = llm.invoke(messages)
            content = response.content.strip()
            
            # Clean markdown code blocks if present
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            elif content.startswith("```"):
                content = content.replace("```", "")
            
            result = json.loads(content)
            
            confidence = result.get("confidence", 0.0)
            filters = result.get("filters", {})
            reasoning = result.get("reasoning", "")
            
            # Remove null/empty values (but keep boolean False as valid)
            cleaned_filters = {
                k: v for k, v in filters.items()
                if v is not None and v != [] and v != "" and v != {}
            }
            
            if confidence >= confidence_threshold and cleaned_filters:
                print(f"🎯 Auto-extracted filters (confidence: {confidence:.2f}): {cleaned_filters}")
                print(f"   Reasoning: {reasoning}")
                return cleaned_filters
            else:
                print(f"⚠️  Low confidence ({confidence:.2f}) for filter extraction, skipping auto-filters")
                return None
                
        except Exception as e:
            print(f"Filter extraction failed: {e}")
            return None
    
    def get_available_categories(self) -> list:
        """Return list of available document categories."""
        return ["Policy", "Circular", "SOP", "Manual", "Report", "Sales Brochure", "Claim Guide", "FAQ", "Other"]

    def get_available_section_types(self) -> list:
        """Return list of available chunk section types."""
        return ["eligibility", "benefits", "financial", "riders", "tax", "claims", "general"]

    def get_available_plan_types(self) -> list:
        """Return list of available plan types."""
        return ["Endowment", "Whole Life", "Money Back", "Term Insurance", "Pension/Annuity", "ULIP", "Child Plan"]
    
    def get_temporal_presets(self) -> Dict[str, Dict[str, str]]:
        """Return common temporal filter presets."""
        now = datetime.now()
        return {
            "last_30_days": {
                "gte": (now - timedelta(days=30)).strftime("%Y-%m-%d")
            },
            "last_90_days": {
                "gte": (now - timedelta(days=90)).strftime("%Y-%m-%d")
            },
            "this_year": {
                "gte": f"{now.year}-01-01"
            },
            "last_year": {
                "gte": f"{now.year - 1}-01-01",
                "lte": f"{now.year - 1}-12-31"
            }
        }
