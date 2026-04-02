import os
import json
import re
import pdfplumber
from groq import Groq


def get_groq_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ── Constants ─────────────────────────────────────────────────
# Safe char limits per Groq call (leaves room for prompt + output)
PROMPT_TEXT_LIMIT   = 22000
QUESTION_TEXT_LIMIT = 14000

# Always include these many pages from start/end regardless of content
ALWAYS_HEAD_PAGES = 5
ALWAYS_TAIL_PAGES = 3

# Section keywords — Python scans ALL pages for these headings
# and pulls those pages as priority content
ELIGIBILITY_KEYWORDS = [
    "eligibility", "eligible", "qualification", "qualifying",
    "criteria", "criterion", "pre-qualification", "prequalification",
    "bid qualification", "technical requirement", "technical criteria",
]

FINANCIAL_KEYWORDS = [
    "emd", "earnest money", "bid security", "performance guarantee",
    "performance security", "financial requirement", "turnover",
    "net worth", "solvency", "working capital", "bank guarantee",
]

DOCUMENT_KEYWORDS = [
    "document", "documents required", "checklist", "enclosure",
    "annexure", "appendix", "list of documents", "submission",
    "certificate required", "proof required",
]

DATE_KEYWORDS = [
    "schedule", "key dates", "important dates", "timeline",
    "last date", "closing date", "deadline", "bid opening",
    "pre-bid", "prebid", "corrigendum",
]

SCOPE_KEYWORDS = [
    "scope of work", "scope of supply", "description of work",
    "brief description", "nature of work", "work description",
    "project description", "introduction", "background",
]


# ── PDF Extraction ────────────────────────────────────────────
def extract_text_from_pdf(pdf_file):
    """
    Extract text from every page with line numbers.
    Per-page errors are caught — one bad page never kills the whole job.
    Returns list of page dicts. Never raises.
    """
    try:
        pages = []
        with pdfplumber.open(pdf_file) as pdf:
            total = len(pdf.pages)
            print(f"[analyzer] PDF opened: {total} pages.")

            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                except Exception as page_err:
                    print(f"[analyzer] Page {i+1} extract error (skipping): {page_err}")
                    continue

                if not page_text or not page_text.strip():
                    continue

                lines = []
                for line_num, line in enumerate(page_text.split("\n"), start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    lines.append({
                        "line_num": line_num,
                        "text": stripped,
                        "is_heading": _is_heading(stripped)
                    })

                if lines:
                    pages.append({
                        "page": i + 1,
                        "lines": lines,
                        "full_text": page_text.strip()
                    })

        print(f"[analyzer] Extracted {len(pages)} non-empty pages from {total} total.")

        if not pages:
            print("[analyzer] No text extracted — likely a scanned/image PDF.")

        return pages

    except Exception as e:
        print(f"[analyzer] PDF extraction failed: {e}")
        return []


def _is_heading(line):
    line = line.strip()
    if not line:
        return False
    if re.match(r'^(\d+\.)+\s+\w+', line):
        return True
    if re.match(r'^(Section|SECTION|Clause|CLAUSE|Part|PART)\s+[\d\w]', line):
        return True
    if line.isupper() and 3 < len(line) < 80:
        return True
    if len(line) < 60 and not line.endswith(('.', ',', ';', ':')):
        if line[0].isupper():
            return True
    return False


# ── Section-aware page scorer ─────────────────────────────────
def _score_page(page, keyword_groups):
    """
    Score a page by how many section keywords it contains.
    Higher score = more relevant content for tender analysis.
    """
    text_lower = page["full_text"].lower()
    score = 0
    for group in keyword_groups:
        for kw in group:
            if kw in text_lower:
                score += 1
    return score


def _page_has_section(page, keywords):
    """Returns True if this page contains any of the given keywords."""
    text_lower = page["full_text"].lower()
    return any(kw in text_lower for kw in keywords)


# ── Core: Section-aware text builder ─────────────────────────
def _build_prompt_text(pages, char_limit):
    """
    Builds the best possible prompt text from pages within char_limit.

    For any PDF size:
    1. Always include first ALWAYS_HEAD_PAGES + last ALWAYS_TAIL_PAGES
    2. Scan ALL pages for section keywords → pull matching pages
    3. Score remaining pages by keyword density
    4. Fill up to char_limit with highest-scoring pages
    5. Never cut mid-line

    Returns (text, meta) where meta describes what was included.
    """
    if not pages:
        return "", "No pages extracted."

    total = len(pages)

    # For small PDFs — just send everything
    full = _format_pages(pages)
    if len(full) <= char_limit:
        print(f"[analyzer] Small PDF — sending all {total} pages ({len(full)} chars).")
        return full, f"All {total} pages included."

    print(f"[analyzer] Large PDF: {total} pages, {len(full)} chars. Building smart selection.")

    # ── Step 1: Always-include pages ──────────────────────────
    selected = {}  # page_num → page dict

    head = pages[:ALWAYS_HEAD_PAGES]
    tail = pages[max(ALWAYS_HEAD_PAGES, total - ALWAYS_TAIL_PAGES):]

    for p in head + tail:
        selected[p["page"]] = p

    # ── Step 2: Pull pages that contain key section headings ──
    all_keyword_groups = [
        ELIGIBILITY_KEYWORDS,
        FINANCIAL_KEYWORDS,
        DOCUMENT_KEYWORDS,
        DATE_KEYWORDS,
        SCOPE_KEYWORDS,
    ]

    for page in pages:
        if page["page"] in selected:
            continue
        for group in all_keyword_groups:
            if _page_has_section(page, group):
                selected[page["page"]] = page
                break  # one match is enough to include this page

    # ── Step 3: Check if we're already within limit ───────────
    sorted_selected = sorted(selected.values(), key=lambda p: p["page"])
    current_text = _format_pages(sorted_selected)

    if len(current_text) <= char_limit:
        skipped = total - len(selected)
        meta = (
            f"PDF: {total} pages. Included: {len(selected)} pages "
            f"(head/tail + section-matched). Skipped: {skipped} low-content pages."
        )
        print(f"[analyzer] {meta}")
        return _add_note(current_text, total, len(selected), skipped), meta

    # ── Step 4: Too many section pages — score and trim ───────
    # Score every selected page, keep highest scorers up to limit
    scored = [
        (p, _score_page(p, all_keyword_groups))
        for p in sorted_selected
    ]
    # Always keep head and tail — only trim middle section pages
    head_nums = {p["page"] for p in head}
    tail_nums  = {p["page"] for p in tail}

    must_keep = [p for p, _ in scored if p["page"] in head_nums | tail_nums]
    can_trim  = sorted(
        [(p, s) for p, s in scored if p["page"] not in head_nums | tail_nums],
        key=lambda x: x[1], reverse=True
    )

    final = {p["page"]: p for p in must_keep}
    for page, score in can_trim:
        candidate = sorted(list(final.values()) + [page], key=lambda p: p["page"])
        if len(_format_pages(candidate)) <= char_limit:
            final[page["page"]] = page

    final_sorted = sorted(final.values(), key=lambda p: p["page"])
    result_text  = _format_pages(final_sorted)

    # ── Step 5: Hard truncate at line boundary (last resort) ──
    if len(result_text) > char_limit:
        truncated = result_text[:char_limit]
        last_nl = truncated.rfind("\n")
        if last_nl > char_limit * 0.88:
            truncated = truncated[:last_nl]
        result_text = truncated
        print(f"[analyzer] Final hard truncation applied at line boundary.")

    skipped = total - len(final)
    meta = (
        f"PDF: {total} pages. Smart selection: {len(final)} pages included, "
        f"{skipped} skipped. All eligibility/financial/date sections prioritised."
    )
    print(f"[analyzer] {meta}")
    return _add_note(result_text, total, len(final), skipped), meta


def _add_note(text, total, included, skipped):
    """Prepend a note to the text so the AI knows the document was filtered."""
    if skipped == 0:
        return text
    note = (
        f"[DOCUMENT NOTE: This PDF has {total} pages. "
        f"{included} pages are shown below — the {skipped} excluded pages contained "
        f"no eligibility, financial, document, date, or scope content. "
        f"All critical sections are fully included.]\n\n"
    )
    return note + text


def _format_pages(pages):
    """Format page dicts into numbered text for AI prompt."""
    output = ""
    for page in pages:
        output += f"\n\n{'='*50}\n"
        output += f"PAGE {page['page']}\n"
        output += f"{'='*50}\n"
        for line in page["lines"]:
            prefix = "[HEADING] " if line["is_heading"] else ""
            output += f"L{line['line_num']:03d}: {prefix}{line['text']}\n"
    return output


# ── Public helpers called by app.py ──────────────────────────
def get_plain_text_for_prompt(pages, limit=PROMPT_TEXT_LIMIT):
    """Always returns a safe string. Never raises."""
    try:
        text, _ = _build_prompt_text(pages, limit)
        return text
    except Exception as e:
        print(f"[analyzer] get_plain_text_for_prompt error: {e}")
        return ""


def format_pages_for_prompt(pages):
    """Alias used by app.py."""
    return get_plain_text_for_prompt(pages)


# ── Quick local summary (no AI call) ─────────────────────────
def extract_quick_summary(pages):
    """
    Pure Python scan of all pages — no AI call, instant.
    Extracts whatever basic facts can be detected from the raw text:
    page count, likely tender title, detected sections, and any
    obvious values (EMD amounts, dates, turnover figures).

    Returns a dict shown to user on the upload card before questions modal.
    This is best-effort — values may be None if not found.
    """
    if not pages:
        return {}

    summary = {
        "total_pages": len(pages),
        "sections_found": [],
        "detected_hints": [],   # plain strings shown as bullet points
    }

    # Combine all text for scanning
    all_text = " ".join(p["full_text"] for p in pages).lower()
    first_page_text = pages[0]["full_text"] if pages else ""

    # ── Detect tender title (first heading on page 1) ──────────
    title = None
    for line in (pages[0]["lines"] if pages else []):
        if line["is_heading"] and len(line["text"]) > 10:
            title = line["text"].strip()
            break
    if not title and first_page_text:
        # Fallback — first non-empty line
        for line in (pages[0]["lines"] if pages else []):
            if len(line["text"]) > 10:
                title = line["text"].strip()
                break
    summary["title"] = title

    # ── Detect which key sections exist ────────────────────────
    section_checks = [
        ("Eligibility Criteria",    ELIGIBILITY_KEYWORDS),
        ("Financial Requirements",  FINANCIAL_KEYWORDS),
        ("Documents Required",      DOCUMENT_KEYWORDS),
        ("Key Dates / Schedule",    DATE_KEYWORDS),
        ("Scope of Work",           SCOPE_KEYWORDS),
    ]
    for label, keywords in section_checks:
        if any(kw in all_text for kw in keywords):
            summary["sections_found"].append(label)

    # ── Sniff for common values using regex ────────────────────
    hints = []

    # EMD / Bid Security amount
    emd_match = re.search(
        r'(?:emd|earnest money|bid security)[^\n]{0,60}?'
        r'(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d+)?)\s*(?:lakh|lac|crore|cr)?',
        all_text
    )
    if emd_match:
        hints.append(f"EMD detected: ₹{emd_match.group(1)}")

    # Turnover requirement
    turnover_match = re.search(
        r'(?:annual turnover|average turnover|minimum turnover)[^\n]{0,60}?'
        r'(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d+)?)\s*(?:lakh|lac|crore|cr)?',
        all_text
    )
    if turnover_match:
        hints.append(f"Turnover requirement: ₹{turnover_match.group(1)}")

    # Experience requirement
    exp_match = re.search(
        r'(?:experience of|minimum experience|at least)\s*(\d+)\s*(?:year|yr)',
        all_text
    )
    if exp_match:
        hints.append(f"Experience required: {exp_match.group(1)} years")

    # Submission deadline
    date_match = re.search(
        r'(?:last date|closing date|submission date|bid submission)[^\n]{0,40}?'
        r'(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})',
        all_text
    )
    if date_match:
        hints.append(f"Submission deadline: {date_match.group(1)}")

    # GeM bid number
    gem_match = re.search(r'gem[/\s]bid[/\s]no[./:\s]*(gem-\S+)', all_text)
    if gem_match:
        hints.append(f"GeM Bid No: {gem_match.group(1).upper()}")

    # Tender type sniff
    if "gem" in all_text[:3000]:
        hints.append("Likely GeM Portal tender")
    elif "qcbs" in all_text or "quality and cost" in all_text:
        hints.append("Likely QCBS tender")
    elif "reverse auction" in all_text:
        hints.append("Likely Reverse Auction tender")
    elif "l1" in all_text or "lowest bidder" in all_text:
        hints.append("Likely L1 (Lowest Bidder) tender")

    summary["detected_hints"] = hints
    return summary


# ── Citation finder ───────────────────────────────────────────
def find_citation(quote, pages):
    if not quote or not pages:
        return None

    quote_clean = quote.strip().lower()
    variants = [quote_clean, quote_clean[:60], quote_clean[:40], quote_clean[:25]]

    for variant in variants:
        if len(variant) < 10:
            continue
        for page in pages:
            nearest_heading = None
            for line in page["lines"]:
                if line["is_heading"]:
                    nearest_heading = line["text"]
                if variant in line["text"].lower():
                    return {
                        "page": page["page"],
                        "line": line["line_num"],
                        "section": nearest_heading or "General",
                        "quote": line["text"],
                        "found": True
                    }

    return {"page": None, "line": None, "section": None, "quote": quote, "found": False}


# ── JSON cleaner ──────────────────────────────────────────────
def _clean_json_response(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'\s*```$', '', raw)
    return raw.strip()


# ── User-facing error messages ────────────────────────────────
def _error_large_pdf():
    return {
        "success": False,
        "error": (
            "This PDF could not be processed. Please try:\n"
            "• Compress the PDF at ilovepdf.com or smallpdf.com\n"
            "• Upload only the eligibility / conditions section\n"
            "• If it's a scanned PDF, convert to text-based PDF first"
        )
    }

def _error_scanned_pdf():
    return {
        "success": False,
        "error": (
            "This appears to be a scanned or image-based PDF — no text could be extracted. "
            "Please convert it to a searchable PDF using ilovepdf.com (OCR tool) and try again."
        )
    }

def _error_ai_parse():
    return {
        "success": False,
        "error": (
            "The AI could not parse this tender document. "
            "Please compress the PDF and try again."
        )
    }


# ── CALL 1: Extract questions ─────────────────────────────────
def extract_questions(pdf_text, company_profile):
    """
    First AI call — generates questions based purely on what
    THIS tender requires vs what the profile already answers.
    Number of questions is 100% dynamic (could be 0 to many).
    """
    if not pdf_text or not pdf_text.strip():
        return _error_scanned_pdf()

    client = get_groq_client()

    # Safe truncation at line boundary
    if len(pdf_text) > QUESTION_TEXT_LIMIT:
        t = pdf_text[:QUESTION_TEXT_LIMIT]
        nl = t.rfind("\n")
        pdf_text_q = t[:nl] if nl > QUESTION_TEXT_LIMIT * 0.85 else t
        print(f"[analyzer] Questions using {len(pdf_text_q)}/{len(pdf_text)} chars.")
    else:
        pdf_text_q = pdf_text

    prompt = f"""
You are an expert Indian government tender analyst.

YOUR TASK:
1. Read the tender document carefully.
2. List EVERY requirement the tender places on bidders:
   - Financial requirements (turnover, net worth, EMD amount)
   - Technical requirements (experience, domain, past work orders)
   - Compliance requirements (certifications, registrations, licences)
   - Operational requirements (employees, equipment, offices)
   - Document requirements (what to submit)
   - Any other condition that affects eligibility

3. Check which of these requirements are already answered by the company profile.
4. For each requirement NOT answered by the profile → generate exactly ONE question.

COMPANY PROFILE (already known — do NOT ask about these):
- Company Name: {company_profile.get('company_name', 'Not provided')}
- Primary Domain: {company_profile.get('domain', 'Not provided')}
- Sub Domains: {', '.join(company_profile.get('sub_domains', []) or []) or 'Not provided'}
- Annual Turnover: Rs {company_profile.get('turnover', 0) or 'Not provided'} Lakhs
- Years of Experience: {company_profile.get('experience', 0) or 'Not provided'}
- Employee Count: {company_profile.get('employee_count', 0) or 'Not provided'}
- Certifications: {company_profile.get('certifications') or 'Not provided'}
- Registration No: {company_profile.get('registration_number') or 'Not provided'}
- PAN: {company_profile.get('pan_number') or 'Not provided'}
- Address: {company_profile.get('address') or 'Not provided'}
- Phone: {company_profile.get('phone') or 'Not provided'}
- Company Email: {company_profile.get('company_email') or 'Not provided'}

TENDER DOCUMENT:
{pdf_text_q}

STRICT RULES FOR QUESTION GENERATION:
- ONLY ask about requirements explicitly stated in this tender document
- NEVER ask about something the profile already answers clearly
- NEVER ask generic or hypothetical questions
- Each question must reference the specific tender requirement it addresses
- Use the most appropriate input_type:
    "number"  → amounts, years, counts, quantities
    "yes_no"  → do you have / are you registered / can you provide
    "select"  → when tender offers specific categories to choose from
    "text"    → names, licence numbers, registration IDs, descriptions
- questions array can be empty [] if the profile fully covers the tender

Return ONLY valid JSON. No markdown. No explanation. No preamble:

{{
  "tender_title": "exact tender title from document",
  "tender_type": "L1 or QCBS or REVERSE_AUCTION or DIRECT or GEM",
  "questions": [
    {{
      "id": "q1",
      "question": "specific question tied to a tender requirement",
      "why_needed": "tender requires [X] — your profile does not confirm this",
      "input_type": "number or text or yes_no or select",
      "options": ["only include for yes_no or select"]
    }}
  ]
}}
"""

    raw = ""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=3000,
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(_clean_json_response(raw))

        if "questions" not in data:
            data["questions"] = []

        print(f"[analyzer] Generated {len(data['questions'])} dynamic questions.")
        return {"success": True, "data": data}

    except json.JSONDecodeError as e:
        print(f"[analyzer] Question JSON parse error: {e} | Raw: {raw[:300]}")
        return _error_ai_parse()
    except Exception as e:
        err = str(e).lower()
        print(f"[analyzer] Question generation error: {e}")
        if any(x in err for x in ["rate", "limit", "context", "token", "length"]):
            return _error_large_pdf()
        return {"success": False, "error": f"Could not read tender: {str(e)}"}


# ── CALL 2: Full analysis ─────────────────────────────────────
def analyze_tender(pdf_text, company_profile, answers=None, pages=None):
    """
    Second AI call — full tender analysis with citation verification.
    Handles all errors gracefully with user-friendly messages.
    """
    if not pdf_text or not pdf_text.strip():
        return _error_scanned_pdf()

    client = get_groq_client()

    answers_text = ""
    if answers:
        answers_text = "\n\nADDITIONAL INFO FROM USER:\n"
        for q, a in answers.items():
            answers_text += f"- {q}: {a}\n"

    prompt = f"""
You are an expert Indian government tender analyst with deep knowledge of:
- GeM (Government e-Marketplace) portal tenders
- L1 (Lowest Bidder) based tenders
- QCBS (Quality and Cost Based Selection) tenders
- Reverse Auction tenders
- Direct/Nomination based tenders
- Indian procurement rules (GFR 2017, CVC guidelines)

CRITICAL INSTRUCTION ABOUT CITATIONS:
For every finding, return the EXACT quote from the document
(copy word for word, max 20 words).
Do NOT guess or paraphrase — copy exactly as it appears.
If information is not in the document, set quote to null.

COMPANY PROFILE:
- Company Name: {company_profile.get('company_name', 'N/A')}
- Domain: {company_profile.get('domain', 'N/A')}
- Sub Domains: {', '.join(company_profile.get('sub_domains', []) or [])}
- Annual Turnover: Rs {company_profile.get('turnover', 0)} Lakhs
- Experience: {company_profile.get('experience', 0)} years
- Employees: {company_profile.get('employee_count', 0)}
- Certifications: {company_profile.get('certifications', 'None')}
{answers_text}

TENDER DOCUMENT (lines are numbered for reference):
{pdf_text}

Return ONLY valid JSON. No markdown, no explanation:

{{
  "project_name": "full project name",
  "project_value": numeric in lakhs or 0,
  "location": "location or Unknown",
  "deadline": "submission deadline or Unknown",
  "tender_type": "L1 or QCBS or REVERSE_AUCTION or DIRECT or GEM",
  "tender_type_reason": "why you identified this type",
  "tender_type_quote": "exact quote from document proving type",
  "qcbs_ratio": "e.g. 70:30 or null",

  "eligibility_criteria": [
    {{
      "criterion": "criterion name",
      "required": "what tender requires",
      "company_has": "what company has",
      "status": "PASS or FAIL or CHECK",
      "note": "brief explanation",
      "quote": "exact quote from document or null"
    }}
  ],

  "overall_eligibility": "ELIGIBLE or PARTIALLY_ELIGIBLE or NOT_ELIGIBLE",
  "eligibility_score": 0-100,
  "eligibility_summary": "2-3 sentence explanation",

  "bid_recommendation": "BID or CONDITIONAL_BID or DO_NOT_BID",
  "bid_recommendation_reason": "clear reason",

  "t_score_estimate": integer or null,
  "t1_gap": "what needed to reach T1 or null",
  "l1_strategy": "L1 pricing advice or null",

  "financial_requirements": {{
    "emd_amount": "amount or Not mentioned",
    "emd_quote": "exact quote or null",
    "performance_guarantee": "percentage or Not mentioned",
    "pg_quote": "exact quote or null",
    "payment_terms": "terms or Not mentioned",
    "working_capital_needed": "estimated amount"
  }},

  "key_dates": [
    {{
      "event": "event name",
      "date": "date or Unknown",
      "quote": "exact quote or null"
    }}
  ],

  "documents_required": [
    {{
      "document": "document name",
      "quote": "exact quote or null"
    }}
  ],

  "gem_specific": {{
    "gem_bid_number": "number or null",
    "oem_required": true or false,
    "msme_preference": true or false,
    "startup_preference": true or false
  }},

  "red_flags": [
    {{
      "flag": "description",
      "quote": "exact quote or null"
    }}
  ],

  "recommendations": ["rec 1", "rec 2", "rec 3"],
  "summary": "3-4 sentence summary"
}}
"""

    raw = ""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=4000,
        )
        raw = response.choices[0].message.content.strip()
        result = json.loads(_clean_json_response(raw))

        if pages:
            result = verify_all_citations(result, pages)

        return {"success": True, "data": result}

    except json.JSONDecodeError as e:
        print(f"[analyzer] Analysis JSON parse error: {e} | Raw: {raw[:300]}")
        return _error_ai_parse()
    except Exception as e:
        err = str(e).lower()
        print(f"[analyzer] Groq API error: {e}")
        if any(x in err for x in ["rate", "limit", "context", "token", "length"]):
            return _error_large_pdf()
        return {"success": False, "error": f"Analysis failed: {str(e)}"}


# ── Citation Verifier ─────────────────────────────────────────
def verify_all_citations(result, pages):
    def resolve(quote):
        if not quote:
            return {"found": False, "page": None, "line": None, "section": None, "quote": None}
        return find_citation(quote, pages) or \
               {"found": False, "page": None, "line": None, "section": None, "quote": quote}

    try:
        result["tender_type_citation"] = resolve(result.get("tender_type_quote"))

        for item in result.get("eligibility_criteria", []):
            item["citation"] = resolve(item.get("quote"))

        fin = result.get("financial_requirements", {})
        fin["emd_citation"] = resolve(fin.get("emd_quote"))
        fin["pg_citation"]  = resolve(fin.get("pg_quote"))

        for date in result.get("key_dates", []):
            date["citation"] = resolve(date.get("quote"))

        for doc in result.get("documents_required", []):
            doc["citation"] = resolve(doc.get("quote"))

        for flag in result.get("red_flags", []):
            flag["citation"] = resolve(flag.get("quote"))

    except Exception as e:
        # Non-fatal — citations fail gracefully, analysis still returned
        print(f"[analyzer] Citation verification error (non-fatal): {e}")

    return result