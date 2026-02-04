"""
3-layer metadata extraction for LIC insurance documents.

Layer 1: Rule-based regex (fast, per-chunk, no LLM)
    - enrich_chunk_metadata(): content pattern matching for tags
    - extract_structured_fields(): plan_number, UIN, ages, amounts, terms

Layer 2: spaCy NER — DISABLED (replaced by enhanced LLM extraction in Layer 3)

Layer 3: LLM-based (per-document, Ollama/Mistral)
    - MetadataExtractor: plan_name, plan_type, summary, target_audience,
      NER entities, date taxonomy, membership types, dynamic fields
"""

import re
import json
from typing import Dict, Any, List, Optional

from app.core.config import settings


# ──────────────────────────────────────────────────────────────────────
# Tag Taxonomy
# ──────────────────────────────────────────────────────────────────────

class LICTagTaxonomy:
    """Comprehensive tag taxonomy for LIC insurance documents."""

    PLAN_TYPES = [
        "Endowment", "Whole Life", "Money Back", "Term Insurance",
        "Pension/Annuity", "ULIP", "Child Plan", "Health Insurance",
        "Micro Insurance", "Group Insurance",
    ]

    PREMIUM_TYPES = [
        "Single Premium", "Limited Pay", "Regular Premium", "Flexible Premium",
    ]

    BENEFIT_TAGS = [
        "Death Benefit", "Maturity Benefit", "Survival Benefit",
        "Guaranteed Additions", "Loyalty Addition", "Bonus (Reversionary)",
        "Terminal Bonus", "Accident Benefit", "Disability Benefit",
        "Critical Illness Cover",
    ]

    FINANCIAL_TAGS = [
        "Tax Benefit (80C)", "Tax Benefit (10(10D))", "Loan Facility",
        "Partial Withdrawal", "Surrender Value", "Paid-up Option", "Auto Cover",
    ]

    RISK_TAGS = [
        "Market Linked", "Non-Linked", "Participating",
        "Non-Participating", "With Profits", "Without Profits",
    ]

    AUDIENCE_TAGS = [
        "Children (0-17)", "Young Adults (18-35)", "Middle Age (36-55)",
        "Senior Citizens (55+)", "Women Specific", "NRI Eligible",
    ]

    GOAL_TAGS = [
        "Child Education", "Child Marriage", "Retirement Planning",
        "Wealth Creation", "Income Protection", "Legacy Planning",
        "Savings", "Investment",
    ]

    SECTION_TAGS = [
        "Eligibility Criteria", "Premium Details", "Benefit Calculation",
        "Death Claim Process", "Maturity Claim Process",
        "Exclusions & Limitations", "Grace Period & Revival",
        "Loan Terms", "Surrender Terms", "Tax Information",
        "Contact & Support", "Grievance Redressal", "Annuity Options",
        "Fund Options", "Rider Details",
    ]


# ──────────────────────────────────────────────────────────────────────
# Layer 1: Rule-based structured field extraction
# ──────────────────────────────────────────────────────────────────────

def extract_structured_fields(text: str) -> Dict[str, Any]:
    """
    Extract structured fields from text using targeted regex.
    Fast, deterministic, no LLM.

    Returns dict with:
        plan_number, uin, age_ranges, monetary_amounts, policy_terms,
        benefit_types, riders_mentioned
    """
    fields: Dict[str, Any] = {}

    # Plan number: "Plan No. 871", "Table No. 936", "Plan 871"
    plan_match = re.search(
        r"(?:plan|table)\s*(?:no\.?|number)\s*[:.]?\s*(\d{2,4})", text, re.IGNORECASE
    )
    if plan_match:
        fields["plan_number"] = plan_match.group(1)

    # UIN: alphanumeric pattern like "512N339V02"
    uin_match = re.search(r"\b(\d{3}[A-Z]\d{3}[A-Z]\d{2})\b", text)
    if uin_match:
        fields["uin"] = uin_match.group(1)

    # Age ranges: "18 to 65 years", "minimum 8 years", "90 days to 55 years"
    age_ranges = []
    for m in re.finditer(
        r"(\d+)\s*(?:to|-)\s*(\d+)\s*(?:years?|yrs?)", text, re.IGNORECASE
    ):
        age_ranges.append(f"{m.group(1)}-{m.group(2)} years")
    # Single ages: "minimum age 18 years"
    for m in re.finditer(
        r"(?:minimum|maximum|min\.?|max\.?|entry)\s*(?:age)?\s*[:.]?\s*(\d+)\s*(?:years?|yrs?)",
        text, re.IGNORECASE,
    ):
        age_ranges.append(f"{m.group(1)} years")
    if age_ranges:
        fields["age_ranges"] = list(set(age_ranges))

    # Monetary amounts: "Rs. 1,00,000", "₹5,00,000", "10 lakh"
    monetary = []
    for m in re.finditer(
        r"(?:rs\.?|₹|inr)\s*([\d,]+(?:\.\d+)?)", text, re.IGNORECASE
    ):
        monetary.append(f"Rs. {m.group(1)}")
    for m in re.finditer(
        r"(\d+(?:\.\d+)?)\s*(?:lakh|crore)", text, re.IGNORECASE
    ):
        unit = "lakh" if "lakh" in m.group(0).lower() else "crore"
        monetary.append(f"{m.group(1)} {unit}")
    if monetary:
        fields["monetary_amounts"] = list(set(monetary))[:10]  # cap at 10

    # Policy terms: "15/20/25 years", "10 to 40 years term"
    terms = []
    for m in re.finditer(
        r"(\d+(?:\s*/\s*\d+)+)\s*(?:years?|yrs?)", text, re.IGNORECASE
    ):
        terms.append(m.group(0).strip())
    for m in re.finditer(
        r"(?:policy\s*term|ppt|premium\s*paying\s*term)\s*[:.]?\s*(\d+)\s*(?:to|-)\s*(\d+)\s*(?:years?|yrs?)",
        text, re.IGNORECASE,
    ):
        terms.append(f"{m.group(1)}-{m.group(2)} years")
    if terms:
        fields["policy_terms"] = list(set(terms))

    # Benefit types detected
    benefit_patterns = {
        "death_benefit": r"death\s*benefit|sum\s*assured\s*on\s*death|risk\s*cover",
        "maturity_benefit": r"maturity\s*benefit|on\s*maturity|vesting\s*benefit",
        "survival_benefit": r"survival\s*benefit|money\s*back|periodic\s*payment",
        "guaranteed_additions": r"guaranteed\s*addition|ga\s*@",
        "loyalty_addition": r"loyalty\s*addition|la\s*@",
        "bonus": r"reversionary\s*bonus|terminal\s*bonus|simple\s*reversionary",
        "annuity": r"annuity\s*(?:option|payable|rate)|immediate\s*annuity|deferred\s*annuity",
    }
    benefit_types = []
    text_lower = text.lower()
    for btype, pat in benefit_patterns.items():
        if re.search(pat, text_lower):
            benefit_types.append(btype)
    if benefit_types:
        fields["benefit_types"] = benefit_types

    # Riders mentioned
    rider_patterns = {
        "ADB": r"\badb\b|accidental\s*death\s*benefit",
        "ATPD": r"\batpd\b|accidental\s*total.*permanent\s*disability",
        "CI": r"\bci\s*rider\b|critical\s*illness\s*rider|new\s*critical\s*illness",
        "WOP": r"\bwop\b|waiver\s*of\s*premium",
        "TAR": r"\btar\b|term\s*assurance\s*rider",
        "Premium Waiver": r"premium\s*waiver\s*(?:benefit|rider)",
    }
    riders = []
    for rider_name, pat in rider_patterns.items():
        if re.search(pat, text_lower):
            riders.append(rider_name)
    if riders:
        fields["riders_mentioned"] = riders

    # Percentages (useful for benefit calculations)
    percentages = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
    if percentages:
        fields["percentages"] = [f"{p}%" for p in percentages[:10]]

    return fields


# ──────────────────────────────────────────────────────────────────────
# Layer 2: spaCy NER extraction — DISABLED
# Replaced by enhanced LLM extraction in Layer 3.
# spaCy required heavy C compilation (blis, thinc) and ~1.5GB VRAM
# for en_core_web_trf. LLM handles NER as part of document-level
# metadata extraction instead.
# ──────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
# Layer 1 (continued): Rule-based per-chunk tag enrichment
# ──────────────────────────────────────────────────────────────────────

# Content-based tag patterns (comprehensive for all LIC document types)
CONTENT_TAG_PATTERNS = {
    # === BENEFIT TYPES ===
    "Death Benefit": r"death\s*benefit|in\s*case\s*of\s*death|life\s*cover|sum\s*assured\s*on\s*death|risk\s*cover|"
                     r"death\s*claim|nominee|risk\s*commenc|death\s*during",
    "Maturity Benefit": r"maturity\s*benefit|on\s*maturity|maturity\s*date|vesting|maturity\s*amount|"
                        r"maturity\s*claim|maturity\s*proceeds|on\s*survival\s*to\s*maturity",
    "Survival Benefit": r"survival\s*benefit|money\s*back|periodic\s*payment|survival\s*amount|"
                        r"survival\s*at\s*end|survival\s*percentage|sb\s*payable",
    "Guaranteed Additions": r"guaranteed\s*addition|ga\s*@|\bga\b.*per\s*annum|guaranteed\s*income|"
                            r"accrued\s*ga|simple\s*ga|guaranteed\s*benefit",
    "Loyalty Addition": r"loyalty\s*addition|la\s*@|\bla\b.*bonus|la\s*at\s*maturity|"
                        r"loyalty\s*bonus|la\s*percentage",
    "Bonus": r"bonus|reversionary|simple\s*reversionary|terminal\s*bonus|final\s*additional\s*bonus|"
             r"fab|interim\s*bonus|accrued\s*bonus|vested\s*bonus|cash\s*bonus",

    # === TAX BENEFITS ===
    "Tax Benefit": r"section\s*80|10\s*\(\s*10\s*d\s*\)|tax\s*(?:benefit|exempt|deduction|free)|income\s*tax\s*act|"
                   r"80c|80ccc|80ccd|tax\s*rebate|tax\s*saving|exempt\s*from\s*tax",

    # === LOAN & SURRENDER ===
    "Loan Facility": r"loan\s*(?:facility|against|available)|policy\s*loan|borrow|"
                     r"loan\s*interest|alf|loan\s*eligibility|outstanding\s*loan",
    "Surrender": r"surrender\s*value|paid.?up|discontinu|special\s*surrender|"
                 r"gsv|ssv|guaranteed\s*surrender|cash\s*value|exit\s*value|"
                 r"reduced\s*paid.?up|acquired\s*paid.?up",
    "Grace Period": r"grace\s*period|revival|lapsed|days\s*grace|"
                    r"reinstatement|revival\s*interest|late\s*payment",

    # === RIDERS & ADDITIONAL BENEFITS ===
    "Premium Waiver": r"premium\s*waiver|waiver\s*of\s*premium|wop|"
                      r"premium\s*exemption|waive\s*future\s*premium",
    "Accident Benefit": r"accident(?:al)?\s*(?:death|benefit)|adb|atpd|accidental\s*death|"
                        r"accidental\s*total|accidental\s*permanent|accident\s*rider",
    "Critical Illness": r"critical\s*illness|ci\s*benefit|dread\s*disease|"
                        r"ci\s*rider|specified\s*illness|listed\s*illness|"
                        r"new\s*critical\s*illness|linked\s*ci|lci",
    "Term Rider": r"term\s*assurance\s*rider|new\s*term|level\s*term|"
                  r"tar|rider\s*sum\s*assured|rider\s*premium",

    # === ANNUITY & PENSION ===
    "Annuity Options": r"annuity\s*option|immediate\s*annuity|deferred\s*annuity|joint\s*life|single\s*life|"
                       r"life\s*annuity|annuity\s*certain|annuity\s*with\s*return|"
                       r"increasing\s*annuity|level\s*annuity|"
                       r"option\s*[a-g]|annuity\s*variant",
    "Pension Plan": r"pension|retirement|annuity|vesting\s*benefit|old\s*age|"
                    r"corpus|accumulation\s*phase|payout\s*phase|"
                    r"jeevan\s*(?:akshay|shanti|nidhi|dhara)|"
                    r"saral\s*pension|pm\s*vaya|atal\s*pension",
    "Commutation": r"commutation|commuted\s*value|one.?third\s*commutation|"
                   r"lump\s*sum\s*at\s*vesting|partial\s*withdrawal\s*at\s*vesting",
    "Deferment": r"deferment\s*period|accumulation\s*period|waiting\s*period|"
                 r"deferred\s*annuity|deferment\s*option|vesting\s*age",

    # === PREMIUM TYPES ===
    "High Sum Assured": r"high\s*sum\s*assured|hsa\s*rebate|rebate\s*on\s*sum|"
                        r"sum\s*assured\s*discount|large\s*sum\s*assured",
    "Limited Pay": r"limited\s*(?:pay|premium)|single\s*premium|one\s*time\s*payment|"
                   r"limited\s*premium\s*paying|5\s*pay|10\s*pay|12\s*pay|15\s*pay",
    "Regular Premium": r"regular\s*premium|annual\s*premium|yearly\s*premium|"
                       r"level\s*premium|throughout\s*term",
    "Flexible Premium": r"flexible\s*premium|variable\s*premium|top.?up|"
                        r"additional\s*premium|extra\s*premium\s*deposit",

    # === PLAN TYPES ===
    "Regular Income": r"regular\s*income|flexi\s*income|income\s*benefit|periodic\s*income|"
                      r"guaranteed\s*income|monthly\s*income|income\s*plan",
    "Whole Life": r"whole\s*life|life\s*long|lifelong\s*cover|"
                  r"till\s*age\s*100|whole\s*of\s*life|entire\s*life",
    "Child Plan": r"child\s*(?:plan|policy|education|marriage)|minor|amritbaal|"
                  r"jeevan\s*tarun|children.*benefit|child.*future|"
                  r"education\s*fund|marriage\s*fund|child\s*deferred",
    "Endowment": r"endowment|jeevan\s*(?:labh|lakshya|anand|umang)|"
                 r"bima\s*(?:ratna|jyoti|shree)|new\s*endowment|"
                 r"savings\s*plan|money\s*back\s*type",
    "Term Insurance": r"term\s*(?:insurance|plan|assurance|cover)|"
                      r"pure\s*protection|risk\s*cover\s*only|"
                      r"tech\s*term|anmol\s*jeevan|jeevan\s*amar|"
                      r"saral\s*jeevan\s*bima|e.?term",
    "Money Back": r"money\s*back|survival\s*benefit|periodic\s*return|"
                  r"jeevan\s*(?:shiromani|tarang)|bima\s*(?:bachat|shree)|"
                  r"new\s*money\s*back|survival\s*percentage",

    # === ULIP SPECIFIC ===
    "ULIP": r"ulip|unit\s*linked|market\s*linked|"
            r"siip|nivesh\s*plus|jeevan\s*(?:umang|profit\s*plus)|"
            r"fund\s*based|nav\s*based|unit\s*value",
    "Fund Options": r"fund\s*option|equity\s*fund|debt\s*fund|balanced\s*fund|"
                    r"bond\s*fund|growth\s*fund|secure\s*fund|"
                    r"money\s*market|liquid\s*fund|index\s*fund",
    "NAV": r"\bnav\b|net\s*asset\s*value|unit\s*price|"
           r"fund\s*value|fund\s*performance|nav\s*date",
    "Fund Switch": r"switch|fund\s*switch|switching\s*charge|"
                   r"transfer\s*between\s*funds|reallocation|rebalancing",

    # === CHARGES (important for ULIPs) ===
    "Mortality Charge": r"mortality\s*charge|cost\s*of\s*insurance|coi|"
                        r"risk\s*charge|mortality\s*deduction",
    "Fund Management Charge": r"fund\s*management\s*charge|fmc|amc|"
                              r"asset\s*management|management\s*fee",
    "Allocation Charge": r"allocation\s*charge|premium\s*allocation|"
                         r"initial\s*allocation|bid.?offer\s*spread",
    "Discontinuance Charge": r"discontinuance\s*charge|surrender\s*charge|"
                             r"early\s*exit|exit\s*charge|discontinuation\s*fund",
    "Admin Charge": r"admin(?:istration)?\s*charge|policy\s*admin|"
                    r"service\s*charge|maintenance\s*charge",

    # === POLICY STATUS & OPERATIONS ===
    "Free Look": r"free\s*look|cooling.?off|cancellation\s*period|"
                 r"return\s*policy|15\s*days|30\s*days\s*period",
    "Assignment": r"assignment|assignee|absolute\s*assignment|"
                  r"conditional\s*assignment|transfer\s*of\s*policy",
    "Nomination": r"nomination|nominee|change\s*of\s*nominee|"
                  r"beneficial\s*nominee|multiple\s*nominee",
    "Policy Loan": r"policy\s*loan|loan\s*against\s*policy|"
                   r"auto\s*loan|alf|loan\s*interest",
    "Revival": r"revival|reinstatement|revive\s*lapsed|"
               r"revival\s*period|revival\s*interest|"
               r"special\s*revival|late\s*revival",
    "Paid-up": r"paid.?up\s*(?:policy|value|status)|"
               r"reduced\s*paid.?up|acquired\s*paid.?up|"
               r"rpv|auto\s*paid.?up",

    # === REGULATORY & COMPLIANCE ===
    "IRDAI": r"irdai|irda|insurance\s*regulatory|"
             r"regulator|regulatory\s*authority|approved\s*by",
    "Section 45": r"section\s*45|non.?disclosure|mis.?statement|"
                  r"contestability|incontestable|"
                  r"repudiation|claim\s*rejection",
    "KYC": r"kyc|know\s*your\s*customer|identity\s*proof|"
           r"address\s*proof|pan\s*card|aadhaar",
    "Exclusions": r"exclusion|not\s*cover|suicide|"
                  r"war|riot|hazardous|aviation|"
                  r"pre.?existing|waiting\s*period",

    # === CLAIM SETTLEMENT ===
    "Claim Process": r"claim\s*(?:process|procedure|settlement)|"
                     r"documents\s*required|claim\s*form|"
                     r"death\s*certificate|proof\s*of\s*(?:death|age)|"
                     r"claim\s*intimation|neft\s*details",
    "Claim Settlement Ratio": r"claim\s*settlement\s*ratio|csr|"
                              r"settlement\s*percentage|claim\s*paid",

    # === PLAN-SPECIFIC IDENTIFIERS ===
    "Jeevan Utsav": r"jeevan\s*utsav|utsav|plan\s*no\.?\s*871",
    "Bima Ratna": r"bima\s*ratna|ratna|plan\s*no\.?\s*863",
    "Jeevan Labh": r"jeevan\s*labh|labh|plan\s*no\.?\s*836",
    "Saral Pension": r"saral\s*pension|plan\s*no\.?\s*862|simple\s*pension",
    "Jeevan Akshay": r"jeevan\s*akshay|akshay|plan\s*no\.?\s*857|immediate\s*annuity",
    "Jeevan Shanti": r"jeevan\s*shanti|shanti|plan\s*no\.?\s*850|deferred\s*annuity",
    "SIIP": r"\bsiip\b|systematic\s*investment|single\s*premium\s*ulip",
    "Nivesh Plus": r"nivesh\s*plus|plan\s*no\.?\s*849|ulip\s*plan",
}

# Section type -> display tag mapping
SECTION_TAG_MAP = {
    "eligibility": "Eligibility Criteria",
    "benefits": "Benefit Details",
    "premium": "Premium Details",
    "sum_assured": "Sum Assured Details",
    "policy_term": "Policy Term Details",
    "exclusions": "Exclusions & Limitations",
    "loan": "Loan Terms",
    "surrender": "Surrender Terms",
    "tax": "Tax Information",
    "rider": "Rider Details",
    "claim": "Claim Process",
    "contact": "Contact & Support",
    "annuity": "Annuity Options",
    "fund": "Fund Options",
    "charges": "Charges & Deductions",
}


def enrich_chunk_metadata(
    chunk_text: str,
    section_type: str = None,
    content_type: str = None,
    doc_metadata: Dict = None,
) -> Dict[str, Any]:
    """
    Enrich a chunk with metadata based on its content.
    Fast, rule-based (no LLM) for chunk-level tagging.

    Combines:
    - Content pattern matching (tags)
    - Structured field extraction (plan_number, UIN, ages, amounts)
    - spaCy NER entities (dates, money, orgs)

    Args:
        chunk_text: The text content of the chunk.
        section_type: Section type from StructuredChunker.
        content_type: Content type ("text", "table").
        doc_metadata: Document-level metadata to inherit.

    Returns:
        Dict with chunk-specific metadata fields.
    """
    # Normalize text for matching
    text_normalized = re.sub(r"\s+", " ", chunk_text)
    text_lower = text_normalized.lower()

    chunk_meta: Dict[str, Any] = {
        "section_type": section_type or "general",
        "content_type": content_type or "text",
        "chunk_tags": [],
        "contains_numbers": False,
        "contains_currency": False,
        "contains_age_info": False,
        "contains_date_info": False,
    }

    # --- Content pattern matching for tags ---
    for tag, pattern in CONTENT_TAG_PATTERNS.items():
        if re.search(pattern, text_lower):
            chunk_meta["chunk_tags"].append(tag)

    # --- Section-based tag ---
    if section_type and section_type in SECTION_TAG_MAP:
        tag = SECTION_TAG_MAP[section_type]
        if tag not in chunk_meta["chunk_tags"]:
            chunk_meta["chunk_tags"].append(tag)

    # --- Boolean flags & entity hints ---
    entity_hints = []

    # Financial content
    if re.search(
        r"₹|rs\.?\s*\d|rupees|\d+,\d{2,3},\d{3}|premium|sum\s*assured|annuity|"
        r"surrender\s*value|fund\s*value|\bnav\b|lakh|crore|\d+(?:\.\d+)?\s*%",
        text_lower,
    ):
        chunk_meta["contains_currency"] = True
        entity_hints.append("Financial")

    # Numerical content
    if re.search(r"\d+(?:\.\d+)?\s*%|\d{1,3}(?:,\d{3})+|\d+\s*(?:lakh|crore)", text_lower):
        chunk_meta["contains_numbers"] = True
        if "Financial" not in entity_hints:
            entity_hints.append("Numerical")

    # Age-related content
    if re.search(
        r"\d+\s*(?:years?|yrs?)|entry\s*age|maturity\s*age|vesting\s*age|"
        r"minimum\s*age|maximum\s*age|policy\s*term|ppt",
        text_lower,
    ):
        chunk_meta["contains_age_info"] = True
        entity_hints.append("Time/Duration")

    # Date content
    if re.search(
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{2}[/-]\d{2}|"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{1,2}",
        text_lower,
    ):
        chunk_meta["contains_date_info"] = True
        if "Time/Duration" not in entity_hints:
            entity_hints.append("Date")

    # Legal/regulatory content
    if re.search(
        r"section\s*\d+|irdai|irda|exclusion|clause|non.?disclosure|"
        r"terms?\s*(?:and|&)\s*conditions?|regulatory|compliance",
        text_lower,
    ):
        entity_hints.append("Legal/Regulatory")

    # Process/procedural content
    if re.search(
        r"claim\s*(?:process|procedure|settlement)|how\s*to|step\s*\d|"
        r"documents?\s*required|procedure|process\s*for",
        text_lower,
    ):
        entity_hints.append("Process/Procedure")

    chunk_meta["entity_hints"] = entity_hints

    # --- Layer 1: Structured field extraction ---
    structured = extract_structured_fields(text_normalized)
    for key, value in structured.items():
        chunk_meta[key] = value

    # --- Layer 2: spaCy NER — DISABLED (handled by LLM in Layer 3) ---
    # NER entities are now extracted at document level via MetadataExtractor
    # and inherited into chunks through doc_metadata.

    # --- Inherit document-level metadata ---
    if doc_metadata:
        # Core identification fields
        for key in ["plan_name", "plan_type", "plan_number", "uin", "category"]:
            if key in doc_metadata and doc_metadata[key]:
                chunk_meta[f"doc_{key}"] = doc_metadata[key]

        # If chunk didn't detect plan_number but doc has it, inherit
        if "plan_number" not in chunk_meta and doc_metadata.get("plan_number"):
            chunk_meta["plan_number"] = doc_metadata["plan_number"]
        if "uin" not in chunk_meta and doc_metadata.get("uin"):
            chunk_meta["uin"] = doc_metadata["uin"]

        # Inherit LLM-extracted NER and date fields from Layer 3
        for key in [
            "organizations", "persons_mentioned",
            "document_date", "effective_date", "notification_date",
            "date_taxonomy", "membership_type", "participation_type",
            "risk_classification", "additional_metadata",
        ]:
            if key in doc_metadata and doc_metadata[key]:
                chunk_meta[key] = doc_metadata[key]

    return chunk_meta


# ──────────────────────────────────────────────────────────────────────
# Layer 3: LLM-based document-level metadata extraction
# ──────────────────────────────────────────────────────────────────────

class MetadataExtractor:
    """
    LLM-based document-level metadata extraction.
    Used once per document (not per chunk). Lazy-loads the LLM.
    """

    def __init__(self):
        self._llm = None
        self.taxonomy = LICTagTaxonomy()

    def _get_llm(self):
        """Lazy-load the LLM for metadata extraction (uses smaller model)."""
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
                # For non-ollama providers, fall back to main LLM
                from app.services.rag.generator import LLMFactory
                generator = LLMFactory.create_generator()
                self._llm = generator.llm

            print(f"  [MetadataExtractor] Using model: {model} ({provider})")
        return self._llm

    def extract_metadata(
        self,
        text: str,
        filename: str,
        headings: Optional[List[str]] = None,
        tables_preview: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract document-level metadata using LLM.

        Sends headings + first 5000 chars + table preview (not 15000 chars of raw text).
        This is more efficient and gives the LLM structured input.
        """
        # Build a focused input for the LLM
        parts = [f"Filename: {filename}"]

        if headings:
            parts.append("Document Headings:\n" + "\n".join(f"- {h}" for h in headings[:30]))

        # First 5000 chars of document text
        parts.append(f"\nDocument Content (first section):\n{text[:5000]}")

        if tables_preview:
            parts.append(f"\nTables found in document:\n{tables_preview[:2000]}")

        user_content = "\n\n".join(parts)

        system_prompt = """You are an expert metadata extractor for Life Insurance Corporation (LIC) of India documents.
Analyze the document and extract structured metadata as JSON.

SECTION 1 — CORE IDENTIFICATION (always extract):
- plan_name: Name of the insurance plan (e.g., "Jeevan Utsav", "Bima Ratna"), null if not a plan document
- plan_number: Plan/Table number (e.g., "871", "943"), null if not found
- uin: Unique Identification Number (e.g., "512N339V02"), null if not found
- plan_type: One of [Endowment, Whole Life, Money Back, Term Insurance, Pension/Annuity, ULIP, Child Plan, Health Insurance, Micro Insurance, Group Insurance], null if unknown
- category: One of [Sales Brochure, Policy Document, Circular, SOP, Manual, Report, Claim Guide, FAQ, Other]
- summary: 1-sentence summary of the plan's key purpose and benefit
- target_audience: Brief description of who this plan is for
- key_benefits: List of 3-5 main benefits/features (strings)
- premium_type: One of [Single Premium, Limited Pay, Regular Premium, Flexible Premium], null if unclear
- keywords: List of 5-7 keywords

SECTION 2 — NAMED ENTITIES (extract all found, empty list if none):
- organizations: List of organizations mentioned (e.g., ["LIC of India", "IRDAI", "LICI"])
- persons_mentioned: List of any named persons (signatories, officials), empty list if none

SECTION 3 — DATE TAXONOMY (extract all dates found, null if not found):
- document_date: Date the document was created/issued (e.g., "2024-01-15"), null if unknown
- effective_date: Date the plan/circular becomes effective, null if not found
- notification_date: Date of IRDAI notification or approval, null if not found
- date_taxonomy: Object with date categories found in the document, e.g.:
  {
    "launch_date": "date the plan was launched",
    "last_modified": "date of last revision",
    "premium_due_dates": "when premiums are due",
    "maturity_dates": "when policy matures",
    "vesting_dates": "when pension/annuity vests",
    "claim_filing_deadline": "deadline to file claims",
    "free_look_period": "cancellation window dates"
  }
  Only include date categories actually present in the document. Values should be the actual dates or descriptions found.

SECTION 4 — MEMBERSHIP & PARTICIPATION:
- membership_type: Type of membership/participation (e.g., "Individual", "Group", "Joint Life", "Family Floater"), null if not applicable
- participation_type: One of ["Participating (with profits)", "Non-Participating", "Unit-Linked", null]
- risk_classification: One of ["Non-Linked Non-Participating", "Non-Linked Participating", "Linked Non-Participating", "Linked Participating", null]

SECTION 5 — DYNAMIC FIELDS (extract any other important metadata you identify):
- additional_metadata: Object with any other important fields you identify from the document that don't fit above categories. Use descriptive keys. Examples:
  {
    "minimum_sum_assured": "Rs. 1,00,000",
    "maximum_sum_assured": "No limit",
    "entry_age_min": "18 years",
    "entry_age_max": "65 years",
    "maturity_age_max": "80 years",
    "policy_term_options": "15/20/25 years",
    "premium_paying_term": "5/10/12 years",
    "loan_available": true,
    "surrender_value_available_after": "3 years"
  }

Return ONLY valid JSON. Do not include markdown formatting or code blocks."""

        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        try:
            llm = self._get_llm()
            response = llm.invoke(messages)
            content = response.content.strip()

            # Clean markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            # Try direct parse first
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Small models often add extra text after JSON.
                # Find the first { and its matching } using a decoder.
                decoder = json.JSONDecoder()
                start = content.find("{")
                if start != -1:
                    obj, _ = decoder.raw_decode(content, start)
                    return obj
                raise
        except Exception as e:
            print(f"LLM metadata extraction failed: {e}")
            # Fall back to rule-based extraction from text
            return self._fallback_extract(text, filename)

    def enrich_chunk_metadata(
        self,
        chunk_text: str,
        section_type: str = None,
        content_type: str = None,
        doc_metadata: Dict = None,
    ) -> Dict[str, Any]:
        """Delegate to module-level enrich_chunk_metadata function."""
        return enrich_chunk_metadata(
            chunk_text=chunk_text,
            section_type=section_type,
            content_type=content_type,
            doc_metadata=doc_metadata,
        )

    def _fallback_extract(self, text: str, filename: str) -> Dict[str, Any]:
        """Rule-based fallback when LLM is unavailable."""
        meta: Dict[str, Any] = {
            # Section 1: Core
            "plan_name": None,
            "plan_number": None,
            "uin": None,
            "plan_type": None,
            "category": "Other",
            "summary": "",
            "keywords": [],
            "key_benefits": [],
            "premium_type": None,
            "target_audience": None,
            # Section 2: NER
            "organizations": ["LIC of India"],
            "persons_mentioned": [],
            # Section 3: Dates
            "document_date": None,
            "effective_date": None,
            "notification_date": None,
            "date_taxonomy": {},
            # Section 4: Membership
            "membership_type": None,
            "participation_type": None,
            "risk_classification": None,
            # Section 5: Dynamic
            "additional_metadata": {},
        }

        fields = extract_structured_fields(text[:5000])
        if fields.get("plan_number"):
            meta["plan_number"] = fields["plan_number"]
        if fields.get("uin"):
            meta["uin"] = fields["uin"]

        # Try to get plan name from filename
        name = filename.replace(".pdf", "").replace("_", " ").replace("-", " ")
        meta["plan_name"] = name

        return meta


# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────

def get_section_display_name(section_type: str) -> str:
    """Convert internal section type to display-friendly name."""
    display_names = {
        "eligibility": "Eligibility Criteria",
        "benefits": "Benefits & Features",
        "premium": "Premium Payment",
        "sum_assured": "Sum Assured",
        "policy_term": "Policy Term",
        "exclusions": "Exclusions & Limitations",
        "loan": "Loan Facility",
        "surrender": "Surrender & Paid-up",
        "tax": "Tax Benefits",
        "rider": "Riders & Add-ons",
        "claim": "Claim Settlement",
        "contact": "Contact Information",
        "annuity": "Annuity Options",
        "fund": "Fund Information",
        "charges": "Charges & Deductions",
        "general": "General Information",
    }
    return display_names.get(section_type, section_type.replace("_", " ").title())
