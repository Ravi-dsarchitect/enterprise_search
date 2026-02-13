"""
Query Guardrail — lightweight LLM classifier that checks whether an incoming
query is within the scope of the project's indexed content before the full
RAG pipeline runs.

Allows: LIC insurance, policies, plans, benefits, claims, premiums, riders,
        tax benefits, circulars, greetings, follow-ups, insurance-adjacent
        finance/tax topics, and any content that could plausibly exist in an
        enterprise document knowledge-base.

Rejects: clearly unrelated topics (cooking, sports, general coding, etc.)
"""

import json
from dataclasses import dataclass
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.rag.generator import LLMFactory


@dataclass
class GuardrailResult:
    """Result of the guardrail check."""
    allowed: bool
    reason: str


# Default rejection message shown to users
GUARDRAIL_REJECTION_MESSAGE = (
    "I'm sorry, but I can only answer questions related to the documents and "
    "information available in this knowledge base (e.g. LIC insurance plans, "
    "policies, benefits, claims, circulars, etc.). "
    "I don't have information to answer this query. "
    "Please try asking something related to the project content."
)

GUARDRAIL_SYSTEM_PROMPT = """\
You are a query classifier for an enterprise document search system.
The system contains documents about Life Insurance Corporation (LIC) of India — insurance plans, policies, benefits, eligibility, premiums, claims, riders, tax benefits, circulars, SOPs, manuals, and related enterprise content.

TASK: Decide if the user's query is IN-SCOPE or OUT-OF-SCOPE.

IN-SCOPE (reply ALLOW):
- Questions about LIC plans, policies, insurance products, benefits, premiums, claims, riders, tax, eligibility, surrender, loan, annuity, pension
- Questions about documents, circulars, SOPs, manuals in the knowledge base
- Greetings, pleasantries, or "hi/hello" messages
- Follow-up questions like "tell me more", "explain further", "what about X?"
- Insurance-adjacent finance or taxation questions
- Any question that could plausibly relate to enterprise / organizational documents

OUT-OF-SCOPE (reply REJECT):
- Cooking recipes, sports scores, entertainment, celebrity gossip
- General programming / coding help unrelated to the system
- Questions about other companies' products (not LIC)
- Creative writing, poetry, stories
- Medical diagnosis, legal advice outside insurance context
- Any topic clearly unrelated to insurance or enterprise documents

Reply with ONLY valid JSON (no markdown):
{"decision": "ALLOW" or "REJECT", "reason": "brief one-line reason"}
"""


class QueryGuardrail:
    """
    Lightweight LLM-based guardrail that classifies queries as in-scope
    or out-of-scope before the RAG pipeline processes them.
    """

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        """Lazily initialise the LLM (re-uses the project's configured provider)."""
        if self._llm is None:
            generator = LLMFactory.create_generator()
            self._llm = generator.llm
        return self._llm

    def check(self, query: str) -> GuardrailResult:
        """
        Classify a query as allowed or rejected.

        Args:
            query: The user's raw query string.

        Returns:
            GuardrailResult with allowed=True/False and a reason.
        """
        if not query or not query.strip():
            return GuardrailResult(allowed=False, reason="Empty query")

        messages = [
            SystemMessage(content=GUARDRAIL_SYSTEM_PROMPT),
            HumanMessage(content=f'Query: "{query}"'),
        ]

        try:
            response = self._get_llm().invoke(messages)
            content = response.content.strip()

            # Clean markdown code blocks if the model wraps its response
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            elif content.startswith("```"):
                content = content.replace("```", "")

            result = json.loads(content.strip())
            decision = result.get("decision", "ALLOW").upper()
            reason = result.get("reason", "")

            allowed = decision != "REJECT"
            print(f"🛡️  Guardrail: {'ALLOW' if allowed else 'REJECT'} — {reason}")
            return GuardrailResult(allowed=allowed, reason=reason)

        except Exception as e:
            # On any failure (JSON parse error, LLM timeout, etc.) we
            # fail-open so the user's query is not blocked.
            print(f"🛡️  Guardrail error (failing open): {e}")
            return GuardrailResult(allowed=True, reason=f"Guardrail error: {e}")
