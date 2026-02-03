from typing import Dict, Any
import json
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.rag.generator import LLMFactory


# LIC Document Tag Taxonomy
class LICTagTaxonomy:
    """Comprehensive tag taxonomy for LIC insurance documents."""

    # Plan Types
    PLAN_TYPES = [
        "Endowment",
        "Whole Life",
        "Money Back",
        "Term Insurance",
        "Pension/Annuity",
        "ULIP",
        "Child Plan",
        "Health Insurance",
        "Micro Insurance",
        "Group Insurance"
    ]

    # Premium Categories
    PREMIUM_TYPES = [
        "Single Premium",
        "Limited Pay",
        "Regular Premium",
        "Flexible Premium"
    ]

    # Benefit Features
    BENEFIT_TAGS = [
        "Death Benefit",
        "Maturity Benefit",
        "Survival Benefit",
        "Guaranteed Additions",
        "Loyalty Addition",
        "Bonus (Reversionary)",
        "Terminal Bonus",
        "Accident Benefit",
        "Disability Benefit",
        "Critical Illness Cover"
    ]

    # Financial Features
    FINANCIAL_TAGS = [
        "Tax Benefit (80C)",
        "Tax Benefit (10(10D))",
        "Loan Facility",
        "Partial Withdrawal",
        "Surrender Value",
        "Paid-up Option",
        "Auto Cover"
    ]

    # Risk Categories
    RISK_TAGS = [
        "Market Linked",
        "Non-Linked",
        "Participating",
        "Non-Participating",
        "With Profits",
        "Without Profits"
    ]

    # Target Audience
    AUDIENCE_TAGS = [
        "Children (0-17)",
        "Young Adults (18-35)",
        "Middle Age (36-55)",
        "Senior Citizens (55+)",
        "Women Specific",
        "NRI Eligible"
    ]

    # Life Goals
    GOAL_TAGS = [
        "Child Education",
        "Child Marriage",
        "Retirement Planning",
        "Wealth Creation",
        "Income Protection",
        "Legacy Planning",
        "Savings",
        "Investment"
    ]

    # Document Sections (for chunk-level tagging)
    SECTION_TAGS = [
        "Eligibility Criteria",
        "Premium Details",
        "Benefit Calculation",
        "Death Claim Process",
        "Maturity Claim Process",
        "Exclusions & Limitations",
        "Grace Period & Revival",
        "Loan Terms",
        "Surrender Terms",
        "Tax Information",
        "Contact & Support",
        "Grievance Redressal",
        "Annuity Options",
        "Fund Options",
        "Rider Details"
    ]


class MetadataExtractor:
    def __init__(self):
        self.llm_generator = LLMFactory.create_generator()
        self.taxonomy = LICTagTaxonomy()

    def extract_metadata(self, text: str, filename: str) -> Dict[str, Any]:
        """
        Uses LLM to extract metadata from the document text.
        Enhanced for comprehensive LIC document tagging.
        """
        # Truncate text to avoid token limits for metadata extraction
        # Increased to 15000 chars to capture more details (benefits, exclusions usually at end)
        sample_text = text[:15000]

        system_prompt = """You are an expert metadata extractor for Life Insurance Corporation (LIC) of India documents.
        Analyze the following document excerpt and extract structured metadata in JSON format.

        REQUIRED FIELDS:
        - title: inferred title of the document
        - summary: 1-sentence summary describing the plan and its key benefit
        - keywords: list of top 5-7 keywords [str]
        - document_date: extracted date if present (YYYY-MM-DD), else null
        - category: classify into [Policy, Circular, SOP, Manual, Report, Sales Brochure, Claim Guide, FAQ, Other]

        LIC PLAN DETAILS (if applicable, else null):
        - plan_name: Name of the insurance plan (e.g., "Jeevan Utsav", "Bima Ratna")
        - plan_number: Plan/Table number if mentioned (e.g., "871", "943")
        - uin: Unique Identification Number if mentioned (e.g., "512N339V02")
        - plan_type: One of [Endowment, Whole Life, Money Back, Term Insurance, Pension/Annuity, ULIP, Child Plan]

        ELIGIBILITY DETAILS:
        - min_entry_age: Minimum entry age (e.g., "8 years", "90 days")
        - max_entry_age: Maximum entry age (e.g., "55 years", "65 years")
        - min_maturity_age: Minimum maturity age if specified
        - max_maturity_age: Maximum maturity age if specified
        - policy_term: Policy term options (e.g., "10-40 years", "15/20/25 years")
        - premium_paying_term: Premium paying term if different from policy term
        - min_sum_assured: Minimum sum assured (e.g., "Rs. 1,00,000")
        - max_sum_assured: Maximum sum assured or "No Limit"

        BENEFIT DETAILS:
        - death_benefit: Brief description of death benefit
        - maturity_benefit: Brief description of maturity benefit
        - survival_benefit: Description if applicable (for Money Back plans)
        - bonus_type: Type of bonus [Reversionary, Terminal, Guaranteed Additions, Loyalty Addition, None]
        - riders_available: List of available riders if mentioned

        COMPREHENSIVE TAGS (select ALL that apply):

        Premium Types:
        - "Single Premium", "Limited Pay", "Regular Premium", "Flexible Premium"

        Benefit Features:
        - "Death Benefit", "Maturity Benefit", "Survival Benefit"
        - "Guaranteed Additions", "Loyalty Addition", "Bonus (Reversionary)", "Terminal Bonus"
        - "Accident Benefit", "Disability Benefit", "Critical Illness Cover"

        Financial Features:
        - "Tax Benefit (80C)", "Tax Benefit (10(10D))", "Loan Facility"
        - "Partial Withdrawal", "Surrender Value", "Paid-up Option", "Auto Cover"

        Risk Category:
        - "Market Linked", "Non-Linked", "Participating", "Non-Participating"

        Target Audience:
        - "Children (0-17)", "Young Adults (18-35)", "Middle Age (36-55)", "Senior Citizens (55+)"
        - "Women Specific", "NRI Eligible"

        Life Goals:
        - "Child Education", "Child Marriage", "Retirement Planning", "Wealth Creation"
        - "Income Protection", "Legacy Planning", "Savings", "Investment"

        Return ONLY valid JSON with all applicable fields.
        """
        
        user_prompt = f"Filename: {filename}\n\nContent:\n{sample_text}"
        
        # We need to access the underlying LLM from the generator to use structured output or just prompt engineering
        # For generality across providers (some don't support json_mode), we'll prompt carefully.
        
        # Accessing the internal LLM from the generator wrapper
        llm = self.llm_generator.llm
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = llm.invoke(messages)
            content = response.content.strip()
            # Basic cleanup if model wraps in markdown code blocks
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            elif content.startswith("```"):
                content = content.replace("```", "")
                
            return json.loads(content)
        except Exception as e:
            print(f"Metadata extraction failed: {e}")
            return {
                "title": filename,
                "category": "Uncategorized",
                "keywords": []
            }

    def enrich_chunk_metadata(self, chunk_text: str, section_type: str = None,
                               content_type: str = None, doc_metadata: Dict = None) -> Dict[str, Any]:
        """
        Enrich a chunk with additional metadata based on its content.
        This is a fast, rule-based method (no LLM) for chunk-level tagging.

        Args:
            chunk_text: The text content of the chunk
            section_type: Section type from LayoutAwareChunker (e.g., "benefits", "eligibility")
            content_type: Content type (e.g., "Table", "List", "Paragraph")
            doc_metadata: Document-level metadata to inherit

        Returns:
            Dictionary with chunk-specific metadata
        """
        import re

        chunk_meta = {
            "section_type": section_type or "general",
            "content_type": content_type or "Paragraph",
            "chunk_tags": [],
            "entity_hints": [],
            "contains_numbers": False,
            "contains_currency": False,
            "contains_age_info": False,
            "contains_date_info": False,
        }

        # Normalize text: collapse multiple whitespace to single space for better matching
        text_normalized = re.sub(r'\s+', ' ', chunk_text)
        text_lower = text_normalized.lower()

        # Detect financial content (comprehensive patterns from all LIC doc types)
        financial_patterns = [
            # Currency symbols and amounts
            r"₹", r"rs\.?\s*\d", r"rupees", r"inr\s*\d",
            r"\d+,\d{2,3},\d{3}", r"\d+,\d{3}",  # Indian number format
            # Premium related
            r"premium", r"tabular\s*premium", r"modal\s*premium",
            r"single\s*premium", r"annual\s*premium",
            # Sum assured
            r"sum\s*assured", r"basic\s*sum", r"death\s*sum",
            r"maturity\s*sum", r"rider\s*sum",
            # Pension/Annuity amounts
            r"annuity", r"corpus", r"purchase\s*price",
            r"annuity\s*payable", r"pension\s*amount",
            # Values
            r"paid.?up\s*value", r"surrender\s*value", r"cash\s*value",
            r"fund\s*value", r"nav", r"unit\s*value",
            # Units
            r"lakh", r"crore", r"per\s*annum", r"p\.?a\.?",
            # Charges (important for ULIPs)
            r"charge", r"fee", r"fmc", r"amc", r"coi",
            r"mortality\s*charge", r"admin\s*charge",
            r"allocation\s*charge", r"discontinuance",
            # Rates and percentages
            r"\d+(?:\.\d+)?\s*%", r"rate\s*of", r"interest\s*rate",
            r"bonus\s*rate", r"annuity\s*rate",
            # Rebates and discounts
            r"rebate", r"discount", r"hsa\s*rebate",
        ]
        if any(re.search(p, text_lower) for p in financial_patterns):
            chunk_meta["contains_currency"] = True
            if "Financial" not in chunk_meta["entity_hints"]:
                chunk_meta["entity_hints"].append("Financial")

        # Detect numerical content (percentages, amounts)
        if re.search(r'\d+(?:\.\d+)?\s*%|\d{1,3}(?:,\d{3})+|\d+\s*(?:lakh|crore)', text_lower):
            chunk_meta["contains_numbers"] = True

        # Detect age-related content (comprehensive patterns from all LIC doc types)
        age_patterns = [
            # Basic age patterns
            r'\d+\s*(?:years?|yrs?)',
            r'age\s*(?:at|of|limit|criteria|proof)',
            # Entry and maturity ages
            r'entry\s*age', r'maturity\s*age', r'vesting\s*age',
            r'minimum\s*age', r'maximum\s*age',
            r'min\.?\s*entry', r'max\.?\s*entry',
            # Age ranges
            r'\d+\s*(?:to|-)\s*\d+\s*years',
            r'between\s*\d+\s*and\s*\d+',
            # Specific age references (common in LIC docs)
            r'age\s*nearer\s*birthday', r'age\s*last\s*birthday',
            r'completed\s*age', r'attained\s*age',
            # Pension-specific ages
            r'vesting\s*(?:age|date)', r'retirement\s*age',
            r'deferment\s*(?:period|age)',
            r'annuity\s*(?:start|commence)',
            # Policy term related
            r'policy\s*term', r'premium\s*paying\s*term', r'ppt',
            r'\d+\s*(?:to|-)\s*\d+\s*(?:years?|yrs?)\s*(?:term|ppt)',
            r'(?:term|duration)\s*(?:of|is)\s*\d+',
            # Child age references
            r'minor', r'child\s*age', r'proposer\s*age',
            r'life\s*assured\s*age',
        ]
        if any(re.search(p, text_lower) for p in age_patterns):
            chunk_meta["contains_age_info"] = True
            if "Age/Duration" not in chunk_meta["entity_hints"]:
                chunk_meta["entity_hints"].append("Age/Duration")

        # Detect policy term content
        term_patterns = [
            r'policy\s*term', r'premium\s*paying\s*term', r'ppt',
            r'\d+\s*years?\s*(?:term|period|duration)'
        ]
        if any(re.search(p, text_lower) for p in term_patterns):
            if "Policy Term" not in chunk_meta["entity_hints"]:
                chunk_meta["entity_hints"].append("Policy Term")

        # Detect date-related content
        if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{2}[/-]\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{1,2}', text_lower):
            chunk_meta["contains_date_info"] = True

        # Section-based tags
        section_tag_map = {
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
        }

        if section_type and section_type in section_tag_map:
            chunk_meta["chunk_tags"].append(section_tag_map[section_type])

        # Content-based tags (whitespace-tolerant patterns for PDF text)
        # Comprehensive patterns from analyzing all LIC document types
        content_patterns = {
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

        for tag, pattern in content_patterns.items():
            if re.search(pattern, text_lower):
                if tag not in chunk_meta["chunk_tags"]:
                    chunk_meta["chunk_tags"].append(tag)

        # Inherit relevant document-level metadata
        if doc_metadata:
            for key in ["plan_name", "plan_type", "plan_number"]:
                if key in doc_metadata and doc_metadata[key]:
                    chunk_meta[f"doc_{key}"] = doc_metadata[key]

        return chunk_meta


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
        "general": "General Information",
    }
    return display_names.get(section_type, section_type.replace("_", " ").title())
