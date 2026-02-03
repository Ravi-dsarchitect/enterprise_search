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
        
        system_prompt = f"""You are a query analysis expert. Extract metadata filter hints from user queries.

Available metadata fields:
- category: [Policy, Circular, SOP, Manual, Report, Sales Brochure, Other]
- source: document filename
- document_date: YYYY-MM-DD format
- keywords: array of relevant terms
- tags: array of relevant tags (e.g. "Tax Benefit", "Single Premium")
- plan_name: e.g. "Jeevan Utsav"
- plan_number: e.g. "871"

Temporal hints:
- "recent", "latest" → use document_date >= {thirty_days_ago}
- "this year" → document_date >= 2026-01-01
- "last year" → document_date between 2025-01-01 and 2025-12-31

LIC hints:
- "Jeevan Utsav" → plan_name: "Jeevan Utsav"
- "Plan 871" → plan_number: "871"
- "Tax benefits of..." → tags: ["Tax Benefit"]

Document type hints:
- "policy", "policies" → category: "Policy"
- "manual", "guide", "documentation" → category: "Manual"
- "SOP", "procedure", "process" → category: "SOP"
- "report", "analysis" → category: "Report"
- "circular", "announcement" → category: "Circular"

Source hints:
- If query mentions specific filename (e.g., "deployment_guide.pdf") → source: that filename

Return JSON with:
{{
  "filters": {{
    "category": "string or null",
    "source": "string or null",
    "document_date": {{"gte": "date", "lte": "date"}} or null,
    "keywords": ["array"] or null,
    "tags": ["array"] or null,
    "plan_name": "string" or null,
    "plan_number": "string" or null
  }},
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}

If no clear filter hints found, return confidence < 0.5 and empty filters.
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
            
            # Remove null/empty values
            cleaned_filters = {k: v for k, v in filters.items() if v is not None and v != [] and v != ""}
            
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
        return ["Policy", "Circular", "SOP", "Manual", "Report", "Other"]
    
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
