# === ROCK-SOLID LLM REFINEMENT (Ollama) =======================================
# Input : df["llm_explanation"]  (your 5-section deterministic text)
# Output: df["llm_refined_explanation"] (analyst-style narrative + Evidence verbatim)
# ==============================================================================

import os, re, json, requests, pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from tqdm import tqdm

import sys, os

# Ensure stdout uses UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    os.environ["PYTHONIOENCODING"] = "utf-8"
def ai_explanation_part_2(df_refined):
    # ---------------- CONFIG ----------------
    ID_COL                 = "base_id_src_po"
    INPUT_COL              = "llm_explanation"
    FINAL_INPUT_EVIDENCE   = "updated_evidence_text"
    VARIANCE_COL           = "variance_summary"
    OUTPUT_COL             = "llm_refined_explanation"
    OLLAMA_URL             = "http://127.0.0.1:11434/api/chat"
    MODEL                  = "llama3:70b"
    
    # Generation knobs (conservative)
    OPTIONS = {
        "temperature": 0.2,
        "top_p": 1,
        "num_ctx": 8192,
        "num_predict": 1400
    }
    
    # Banned terms (post-filter)
    BANNED_TERMS = [
        r"\bfraud\b", r"\bfraudulent\b", r"\bscam\b", r"\bscamming\b",
        r"\bcorruption\b", r"\bbribe\b", r"\bbribery\b"
    ]
    HEADER_LINE = "_" * 152
    
    # Checkpointing (optional)
    CHECKPOINT_EVERY = 0
    CHECKPOINT_PATH  = "with_llm_refined_explanations_checkpoint_08_09.parquet"
    
    # ---------------- SMALL PARSERS (robust for your format) ----------------
    def _find_block(src: str, title_regex: str) -> str:
        """
        Return lines AFTER a heading that matches title_regex (numbered and/or bold) until next heading or EOF.
        Accepts lines like: **2. Key Description (PO Details)**  or  2. Key Description  or  **Risk Drivers**
        """
        src = src or ""
        lines = src.splitlines()
        # heading: optional **, optional "n.", the title_regex, optional "(...)", optional **, and only whitespace around
        head_pat = re.compile(rf"^\s*\*{{0,2}}\s*(?:\d+\.\s*)?{title_regex}(?:\s*\([^)]+\))?\s*\*{{0,2}}\s*$", re.I)
        start = None
        for i, ln in enumerate(lines):
            if head_pat.match(ln.strip()):
                start = i + 1
                break
        if start is None:
            return ""
        # next heading detector (very forgiving)
        next_head_pat = re.compile(r"^\s*\*{0,2}\s*(?:\d+\.\s*)?[A-Z][A-Za-z0-9 &/()\-]+(?:\s*&\s*[A-Z][^)]*)?\s*\*{0,2}\s*$")
        out = []
        for ln in lines[start:]:
            if next_head_pat.match(ln.strip()):
                break
            out.append(ln)
        while out and out[-1].strip() == "":
            out.pop()
        return "\n".join(out)
    
    def extract_risk_line(src: str) -> str:
        ctx = _find_block(src, r"Context\s*&\s*Trigger")
        for ln in ctx.splitlines():
            if "Risk Score" in ln:
                return ln.strip()
        return ""
    
    def extract_primary_line(src: str) -> str:
        ctx = _find_block(src, r"Context\s*&\s*Trigger")
        for ln in ctx.splitlines():
            if "Primary Risk Scenario" in ln:
                return ln.strip()
        return ""
    
    def extract_risk_drivers(src: str) -> list[str]:
        """
        Your drivers are plain lines (not bullets). Accept any non-empty trimmed line.
        """
        blk = _find_block(src, r"Risk\s*Drivers")
        drivers = []
        for ln in blk.splitlines():
            t = ln.strip().lstrip("•-*· ").rstrip(". ")
            if t:
                drivers.append(t)
        return drivers
    
    def extract_business_impact_line(src: str) -> str:
        blk = _find_block(src, r"Business\s*Impact")
        for ln in blk.splitlines():
            t = ln.strip()
            if t and ("Suspicious" in t or "Line Value" in t or "Flagged Value" in t or "Overall Impact" in t):
                return t
        for ln in blk.splitlines():
            if ln.strip():
                return ln.strip()
        return ""
    
    def source_has_no_risk(source_text: str) -> bool:
        return bool(re.search(r"\bNo\s*Risk\b", source_text or "", re.IGNORECASE))
    
    # ---------------- Price Variance (your exact format) ----------------
    def _normalize(s: str) -> str:
        if s is None: return ""
        s = str(s)
        s = s.replace("—", "-").replace("–", "-").replace("₹", "INR ")
        s = re.sub(r"[ \t]+", " ", s)
        return s.strip()
    
    def _first(v: str, patterns: list[str]):
        for p in patterns:
            m = re.search(p, v, flags=re.I)
            if m: return m.group(1).strip()
        return None
    
    def _clean_arrow(s: str | None) -> str | None:
        if not s: return None
        t = re.sub(r"^[•\-\u2022>\u2192]+\s*", "", s.strip())
        return f"• {t}"
    
    def build_price_variance_bullets(variance_text: str) -> str:
        v = _normalize(variance_text)
        curr  = _first(v, [r"Current\s*Price\s*(?:Per\s*Unit|/Unit)\s*:\s*([^|;]+)"])
        avg   = _first(v, [r"Avg(?:erage)?\s*Price\s*(?:Per\s*Unit|/Unit)(?:\s*\([^)]+\))?\s*:\s*([^|;]+)"])
        delta = _first(v, [r"[Δ∆]\s*/?\s*Unit\s*:\s*([^|;]+)", r"Variance\s*Value\s*Per\s*Unit\s*:\s*([^|;]+)"])
        total = _first(v, [r"Total\s*Variance\s*Value\s*:\s*([^|;]+)"])
        arrow = _clean_arrow(_first(v, [r"(→\s*[^|;]+)"])) or "• Current PO price is slightly below/high then average price."
        return "\n".join([
            f"• Current Price Per Unit: {curr or 'None'}",
            "• Average Price Per Unit (Same and Different Vendor Compared Transactions from the compared transactions examples): " + (avg or 'None'),
            f"• Variance Value Per Unit: {delta or 'None'}",
            arrow,
            "• Total Variance Value: " + (total or 'None') + " {Value = (Current price per unit - Average price per unit) x Current Qty}"
        ])
    
    def insert_price_variance_block(narr: str, variance_block: str) -> str:
        hdr = re.compile(r'(?im)^\s*\*{0,2}\s*Recommended\s+Next\s+Steps\s*\*{0,2}\s*$', re.M)
        m = hdr.search(narr)
        block = f"\n**Price Variance Value**\n{variance_block}\n"
        if m:
            pos = m.start()
            return narr[:pos] + block + narr[pos:] + "\n"
        return narr + block
    
    # ---------------- PROMPT ----------------
    def build_refine_prompt(source_text: str) -> str:
        no_risk = source_has_no_risk(source_text)
        no_risk_note = (
            "- If the SOURCE indicates 'No Risk', keep it minimal:\n"
            "  • Why It Was Flagged: 'SARA has not flagged this transaction.'\n"
            "  • Recommended Next Steps: 1–2 light bullets (retain documentation / periodic review).\n"
        )
        return f"""
    You are a senior procurement risk analyst. Use ONLY the SOURCE (verbatim tokens). Do NOT invent data.
    Do NOT include any section titled 'Price Variance Value' (the caller inserts it).
    For Executive Summary strictly pick up data from Key Description section only. Pick data corresponding to those value asked only.
    DO NOT hallucinate. For Risk Flag pick data strictly from Context & Trigger Risk Scenario only.
    HARD CONSTRAINTS:
    - Avoid legal/accusatory terms (fraud, bribery, etc.).
    - Preserve IDs and codes exactly if you reference them.
    - If a value is not present in SOURCE, write "(not available)".
    - Give OUTPUT IN THE EXACT FORMAT ONLY. No extra text outside the template.
    STRICT OUTPUT FORMAT (exact headers, bullets as shown, no extra sections):
    **SARA AI — Explanation**
    
    **Executive Summary**
    • Purchase Order: <'PO <po> / Item <item>' from 'PO / Item & PR Ref' (ignore anything after '&') or '(not available)'>
    • Vendor: <'code – name' from 'Vendor no - Name' or '(not available)'>
    • Material: <'code — text' from 'Material – Text, Type' (text before comma) or '(not available)'>
    • Risk Flag: <from 'Risk Scenario' (include sub-risks if listed) or '(not available)'>
    • Line Value: <copy 'Flagged Value: ...' or 'Net / Gross Value' or compute '<currency> <qty*unit_price, 2dp>' using ONLY numbers in SOURCE; else '(not available)'>
    • Action: <short phrase from Context & Trigger like 'Needs Validation'/'High Risk' etc., else '(not available)'>
    
    **Why It Was Flagged**
    • <bullet grounded strictly in SOURCE (only if present)>
    • <bullet grounded strictly in SOURCE (only if present)>
    • <bullet grounded strictly in SOURCE (only if present)>
    • <bullet grounded strictly in SOURCE (only if present)>
    
    **Recommended Next Steps**
    1. <specific, actionable step grounded in SOURCE>
    2. <specific, actionable step grounded in SOURCE>
    3. <specific, actionable step grounded in SOURCE>
    
    {no_risk_note if no_risk else ""}
    
    SOURCE (read-only; copy tokens; do not reformat)
    — BEGIN —
    {source_text}
    — END —
    """.strip()
    
    # ---------------- OLLAMA CLIENT (persistent session + retries) ----------------
    class OllamaClient:
        def __init__(self, base_url: str, model: str, options: dict):
            self.base_url = base_url.rstrip("/")
            self.model = model
            self.options = options or {}
            self.sess = requests.Session()
            self.sess.headers.update({"Content-Type": "application/json"})
    
        def _payload(self, prompt: str) -> dict:
            return {
                "model": self.model,
                "messages": [
                    {"role": "system", "content":
     "Be precise and conservative. Never invent data or numbers. Do not output an Evidence section. "
     "Do not include numeric risk score in the Executive Summary; classification name may be used. "
     "If mentioning a PO, include its item number (e.g., 'PO 4000003611 / Item 00010'). "
     "Avoid legal/accusatory terms (risk language only)." },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": self.options
            }
    
        @retry(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential_jitter(1, 6),
            retry=retry_if_exception_type((requests.RequestException,))
        )
        def chat(self, prompt: str, timeout: int = 600) -> str:
            r = self.sess.post(self.base_url, data=json.dumps(self._payload(prompt)), timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return (data.get("message", {}) or {}).get("content", "") or ""
    
    # ---------------- SANITIZERS & LOCAL FALLBACKS --------------------------------
    def sanitize_banned_terms(text: str) -> str:
        out = text or ""
        for pat in BANNED_TERMS:
            out = re.sub(pat, "risk", out, flags=re.IGNORECASE)
        return out
    
    def local_narrative_fallback(source_text: str) -> str:
        risk_line = extract_risk_line(source_text)
        primary   = extract_primary_line(source_text)
        impact    = extract_business_impact_line(source_text)
        drivers   = extract_risk_drivers(source_text)
    
        exec_bits = []
        if risk_line: exec_bits.append(risk_line)
        if primary:   exec_bits.append(primary)
        if impact:    exec_bits.append(impact)
        exec_summary = " ".join(exec_bits) or "This transaction was evaluated; see Evidence for details."
    
        why_bullets = [f"· {d}" for d in (drivers[:4] or ["No specific risk drivers listed in the source."])]
    
        next_steps = [
            "· Validate price justification and contract terms for this PO line.",
            "· Cross-check material/vendor pricing against recent comparable lines.",
            "· Confirm requester acknowledgment of any variance and approval trail."
        ]
    
        return (
            "**SARA AI — Explanation**\n\n"
            "**Executive Summary**\n" + exec_summary + "\n\n"
            "**Why It Was Flagged**\n" + "\n".join(why_bullets) + "\n\n"
            "**Recommended Next Steps**\n" + "\n".join(next_steps)
        )
    
    # ---------------- SECTION MATCHERS & HEADER LINE INSERTION ---------------------
    SECTION_PATTERNS = [
        re.compile(r'^\s*\*{0,2}\s*SARA\s*AI\s*[—]?\s*Explanation\s*\*{0,2}\s*$', re.I),
        re.compile(r'^\s*\*{0,2}\s*Executive\s+Summary\s*\*{0,2}\s*$', re.I),
        re.compile(r'^\s*\*{0,2}\s*Why\s+It\s+Was\s+Flagged\s*\*{0,2}\s*$', re.I),
        re.compile(r'^\s*\*{0,2}\s*Price\s+Variance(?:\s+Value)?\s*\*{0,2}\s*$', re.I),
        re.compile(r'^\s*\*{0,2}\s*Recommended\s+Next\s+Steps\s*\*{0,2}\s*$', re.I),
    ]
    
    def add_header_line_before_sections(text: str, header_line: str = HEADER_LINE, times: int = 1) -> str:
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        lines = text.splitlines(True)
        i = 0
        while i < len(lines):
            line_wo_nl = lines[i].rstrip("\n")
            if any(p.match(line_wo_nl) for p in SECTION_PATTERNS):
                j = i - 1
                while j >= 0 and lines[j].strip() == "":
                    j -= 1
                already_has = (j >= 0 and lines[j].strip() == header_line)
                if not already_has:
                    lines.insert(i, (header_line + "\n") * times)
                    i += 1
            i += 1
        return "".join(lines)
    
    # ---------------- VALIDATION & REPAIR GUARDRAILS -------------------------------
    EXEC_BULLETS = [
        r'•\s*Purchase\s+Order:\s*',
        r'•\s*Vendor:\s*',
        r'•\s*Material:\s*',
        r'•\s*Risk\s+Flag:\s*',
        r'•\s*Line\s+Value:\s*',
        r'•\s*Action:\s*'
    ]
    
    def _has_section(txt: str, title: str) -> bool:
        return re.search(rf'(?mi)^\s*\*\*{re.escape(title)}\*\*\s*$', txt or "") is not None
    
    def _section_block(txt: str, title: str,
                       next_titles=('Why It Was Flagged','Price Variance','Price Variance Value','Recommended Next Steps')) -> str:
        if not txt: return ""
        pat = re.compile(
            rf'(?is)^\s*\*\*{re.escape(title)}\*\*\s*\n(.*?)(?=\n\s*\*\*(?:{"|".join(map(re.escape,next_titles))})\*\*|\Z)',
            re.M
        )
        m = pat.search(txt)
        return m.group(1).strip() if m else ""
    
    def _count_bullets(block: str, bullet='•') -> int:
        return len(re.findall(rf'(?m)^\s*{re.escape(bullet)}\s+', block or ""))
    
    def _count_numbered(block: str) -> int:
        return len(re.findall(r'(?m)^\s*\d+\.\s+', block or ""))
    
    def is_valid_narrative(txt: str) -> bool:
        if not txt: return False
        if not (_has_section(txt, "SARA AI — Explanation") and
                _has_section(txt, "Executive Summary") and
                _has_section(txt, "Why It Was Flagged") and
                _has_section(txt, "Recommended Next Steps")):
            return False
    
        exec_blk = _section_block(txt, "Executive Summary")
        if not exec_blk: return False
        for rb in EXEC_BULLETS:
            if re.search(rb, exec_blk) is None:
                return False
    
        why_blk = _section_block(txt, "Why It Was Flagged")
        if _count_bullets(why_blk) < 1:
            return False
    
        steps_blk = _section_block(txt, "Recommended Next Steps")
        if _count_numbered(steps_blk) < 3:
            return False
        return True
    
    def strip_preamble_noise(txt: str) -> str:
        if not txt: return ""
        m = re.search(r'(?ims)^\s*\*\*SARA\s*AI\s*[—\-–]\s*Explanation\*\*.*', txt)
        return m.group(0).strip() if m else txt.strip()
    
    # ---------------- Deterministic builder (from SOURCE) --------------------------
    # Parse Markdown rows like: | label | value |
    MD_ROW_PAT = re.compile(r'^\s*\|\s*(?P<label>[^|]+?)\s*\|\s*(?P<val>[^|]*?)\s*\|\s*$', re.M)
    
    def _md_val(block: str, label_variants: list[str]) -> str | None:
        if not block: return None
        want = {lv.strip().lower() for lv in label_variants}
        for m in MD_ROW_PAT.finditer(block):
            lab = (m.group("label") or "").strip()
            if lab.lower() in ("field", "---"):   # skip header/separator
                continue
            if lab.strip().lower() in want:
                return (m.group("val") or "").strip()
        return None
    
    def _kv_val_anywhere(src: str, label_variants: list[str]) -> str | None:
        if not src: return None
        for lab in label_variants:
            m = re.search(rf'(?mi)^\s*{re.escape(lab)}\s*[:|]\s*(.+?)\s*$', src)
            if m:
                return m.group(1).strip()
        return None
    
    def _parse_qty_price(s: str):
        # "5 EA @ INR 144.07"
        if not s: return (None, None, None, None)
        m = re.search(r"([\d,]+(?:\.\d+)?)\s*([A-Za-z%]+)?\s*@\s*([A-Za-z]{3})\s*([\d,]+(?:\.\d+)?)", s)
        if not m: return (None, None, None, None)
        qty  = float(m.group(1).replace(",", "")) if m.group(1) else None
        unit = m.group(2) or None
        cur  = m.group(3) or None
        ppu  = float(m.group(4).replace(",", "")) if m.group(4) else None
        return qty, unit, ppu, cur
    
    def _num(s: str) -> float | None:
        if not s: return None
        m = re.search(r"([\d,]+(?:\.\d+)?)", s)
        return float(m.group(1).replace(",", "")) if m else None
    
    def _fmt_money(cur: str | None, amt: float | None) -> str | None:
        if cur and amt is not None: return f"{cur} {amt:,.2f}"
        if amt is not None: return f"{amt:,.2f}"
        return None
    
    def _build_exec_from_source(src: str) -> str:
        kd = _find_block(src, r"Key\s*Description")  # accepts "(PO Details)" automatically
        po_item_pr = _md_val(kd, ["PO / Item & PR Ref"]) or _kv_val_anywhere(src, ["PO / Item & PR Ref"])
        vendor     = _md_val(kd, ["Vendor no - Name"])   or _kv_val_anywhere(src, ["Vendor no - Name"])
        material   = _md_val(kd, ["Material – Text, Type", "Material - Text, Type", "Material – Text"]) \
                     or _kv_val_anywhere(src, ["Material – Text, Type", "Material - Text, Type", "Material – Text"])
        qtyprice   = _md_val(kd, ["Quantity / Unit & Unit Price"]) or _kv_val_anywhere(src, ["Quantity / Unit & Unit Price"])
        netgross   = _md_val(kd, ["Net / Gross Value", "Flagged Value", "Net Value", "Gross Value"]) \
                     or _kv_val_anywhere(src, ["Net / Gross Value", "Flagged Value", "Net Value", "Gross Value"])
    
        ctx = _find_block(src, r"Context\s*&\s*Trigger") or src
        risk_flag = None
        for ln in (ctx or "").splitlines():
            if "Risk Scenario" in ln:
                risk_flag = ln.split(":", 1)[-1].strip()
                break
    
        action = None
        for ln in (ctx or "").splitlines():
            if "Risk Score" in ln and "→" in ln:
                action = ln.split("→", 1)[-1].strip()
                break
    
        # Purchase Order / Item
        po_item = "(not available)"
        if po_item_pr:
            m = re.search(r"(\d{8,})\s*/\s*(\d{5})", po_item_pr)
            if m: po_item = f"PO {m.group(1)} / Item {m.group(2)}"
    
        # Material "code — text(before comma)"
        material_fmt = "(not available)"
        if material:
            m = re.search(r"(\d+)\s*[—\-]\s*([^,|]+)", material)
            material_fmt = f"{m.group(1)} — {m.group(2).strip()}" if m else material
    
        # Line Value: prefer provided; else qty*ppu
        qty, unit, ppu, cur = _parse_qty_price(qtyprice or "")
        line_val = None
        if netgross:
            m = re.search(r"([A-Za-z]{3})\s*([\d,]+(?:\.\d+)?)", netgross)
            if m:
                line_val = f"{m.group(1)} {float(m.group(2).replace(',','')):,.2f}"
            else:
                ng = _num(netgross)
                line_val = _fmt_money(cur, ng) if ng is not None else netgross
        if not line_val and (qty is not None and ppu is not None):
            line_val = _fmt_money(cur, qty * ppu)
    
        return "\n".join([
            f"• Purchase Order: {po_item or '(not available)'}",
            f"• Vendor: {vendor or '(not available)'}",
            f"• Material: {material_fmt or '(not available)'}",
            f"• Risk Flag: {risk_flag or '(not available)'}",
            f"• Line Value: {line_val or '(not available)'}",
            f"• Action: {action + ' go through evidence for further validation' or '(not available)'}",
        ])
    
    def _build_why_from_source(src: str) -> str:
        ds = extract_risk_drivers(src) or []
        if not ds: return "• No specific risk drivers listed in the source."
        return "\n".join([f"• {d}" for d in ds[:4]])
    
    def _build_steps_from_source(src: str) -> str:
        return "\n".join([
            "1. Validate price justification and contract terms for this PO line.",
            "2. Cross-check material/vendor pricing against recent comparable lines.",
            "3. Confirm requester approval trail for any acceptable variance."
        ])
    
    def build_doc_from_source(src: str) -> str:
        return (
            "**SARA AI — Explanation**\n\n"
            "**Executive Summary**\n" + _build_exec_from_source(src) + "\n\n"
            "**Why It Was Flagged**\n" + _build_why_from_source(src) + "\n\n"
            "**Recommended Next Steps**\n" + _build_steps_from_source(src)
        )
    
    def repair_to_format_with_llm(source_text: str, bad_text: str) -> str:
        client_fix = OllamaClient(OLLAMA_URL, MODEL, {**OPTIONS, "temperature": 0.0})
        repair_prompt = f"""
    Reformat STRICTLY into the EXACT template below using ONLY values present in SOURCE.
    If a value is missing, write "(not available)". No extra text outside the template. No "Price Variance Value" section.
    
    TEMPLATE:
    **SARA AI — Explanation**
    
    **Executive Summary**
    • Purchase Order: <value>
    • Vendor: <value>
    • Material: <value>
    • Risk Flag: <value>
    • Line Value: <value>
    • Action: <value>
    
    **Why It Was Flagged**
    • <value>
    • <value>
    • <value>
    • <value>
    
    **Recommended Next Steps**
    1. <value>
    2. <value>
    3. <value>
    
    SOURCE
    — BEGIN —
    {source_text}
    — END —
    
    CURRENT OUTPUT
    — BEGIN —
    {bad_text}
    — END —
    """.strip()
        try:
            fixed = client_fix.chat(repair_prompt)
            return strip_preamble_noise(fixed)
        except Exception:
            return ""
    
    # ---------------- ENSURE STRUCTURE (no placeholders) ---------------------------
    def ensure_sections_present(text: str, source_fallback: str) -> str:
        t = (text or "").strip()
        if not t:
            return build_doc_from_source(source_fallback)
        has_all = (_has_section(t, "SARA AI — Explanation") and
                   _has_section(t, "Executive Summary") and
                   _has_section(t, "Why It Was Flagged") and
                   _has_section(t, "Recommended Next Steps"))
        return t if has_all else build_doc_from_source(source_fallback)
    
    # ---------------- DRIVER -------------------------------------------------------
    def refine_with_llm(df_in: pd.DataFrame,
                        input_col: str = INPUT_COL,
                        id_col: str = ID_COL,
                        variance_col: str = VARIANCE_COL,
                        output_col: str = OUTPUT_COL,
                        checkpoint_every: int = CHECKPOINT_EVERY,
                        checkpoint_path: str = CHECKPOINT_PATH,
                        final_evidence: str = FINAL_INPUT_EVIDENCE) -> pd.DataFrame:
        df2 = df_in.copy()
        outs = []
    
        client = OllamaClient(OLLAMA_URL, MODEL, OPTIONS)
    
        for i, row in tqdm(df2.iterrows(), total=len(df2), desc="Refining with LLM"):
            main_source = str(row.get(final_evidence, "") or "")
            source      = str(row.get(input_col, "") or "").strip()
            if not source and main_source:
                source = main_source
    
            variance_text = str(row.get(variance_col, "") or "").strip()
            variance_block = build_price_variance_bullets(variance_text)
    
            if not source:
                outs.append("")   # keep empty string, never NaN
            else:
                prompt = build_refine_prompt(source)
                try:
                    narrative = client.chat(prompt)
                except Exception:
                    narrative = ""
    
                narrative = strip_preamble_noise(narrative)
                if not narrative.strip():
                    narrative = local_narrative_fallback(source)
    
                if not is_valid_narrative(narrative):
                    repaired = repair_to_format_with_llm(source, narrative)
                    if repaired and is_valid_narrative(repaired):
                        narrative = repaired
                    else:
                        narrative = build_doc_from_source(source)
    
                final_doc = insert_price_variance_block(narrative, variance_block) if 'NaN' not in variance_block else narrative
                final_doc = sanitize_banned_terms(final_doc)
                final_doc = ensure_sections_present(final_doc, source)
                final_doc = add_header_line_before_sections(final_doc, times=1)
    
                final_text = final_doc.rstrip() + "\n" + main_source
                outs.append(final_text)
    
            #if checkpoint_every and (i + 1) % checkpoint_every == 0:
                #tmp = df2.iloc[: i + 1].copy()
                #tmp[output_col] = pd.Series(outs, index=tmp.index, dtype="string")
                #tmp.to_parquet(checkpoint_path, index=False)
    
        df2[output_col] = pd.Series(outs, index=df2.index, dtype="string").fillna("")
        return df2
    
    # ---------------- RE-RUN UNTIL CLEAN (remove 'See Evidence') -------------------
    SEE_EVIDENCE_RE = r'(?is)\bsee\s*evidence\b\.?'
    
    def _needs_fix_mask(df: pd.DataFrame) -> pd.Series:
        col = OUTPUT_COL
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found.")
        return df[col].fillna("").str.contains(SEE_EVIDENCE_RE, regex=True)
    
    def _read_any(path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".parquet":
            return pd.read_parquet(path)
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
        elif ext == ".csv":
            return pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported input extension: {ext}")
    
    def _write_any(df: pd.DataFrame, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".parquet":
            df.to_parquet(path, index=False)
        elif ext in (".xlsx", ".xls"):
            df.to_excel(path, index=False)
        elif ext == ".csv":
            df.to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported output extension: {ext}")
    
    def _update_master_from_fixed(master: pd.DataFrame, fixed: pd.DataFrame) -> None:
        if ID_COL not in master.columns or ID_COL not in fixed.columns:
            raise KeyError(f"Both dataframes must contain '{ID_COL}'.")
        if OUTPUT_COL not in fixed.columns:
            raise KeyError(f"Fixed dataframe must contain '{OUTPUT_COL}'.")
        new_map = fixed.set_index(ID_COL)[OUTPUT_COL]
        sel = master[ID_COL].isin(new_map.index)
        master.loc[sel, OUTPUT_COL] = master.loc[sel, ID_COL].map(new_map)
    
    def rerun_llm_until_clean(df_refined,max_passes: int = 2):
        #df_all = _read_any(in_path)
        df_all=df_refined
    
        for p in range(1, max_passes + 1):
            mask = _needs_fix_mask(df_all)
            n = int(mask.sum())
            print(f"[Pass {p}/{max_passes}] rows needing re-run: {n}")
            if n == 0:
                break
    
            subset_cols = [ID_COL, INPUT_COL, VARIANCE_COL, FINAL_INPUT_EVIDENCE]
            missing = [c for c in subset_cols if c not in df_all.columns]
            if missing:
                raise KeyError(f"Missing required column(s) in input: {missing}")
    
            subset = df_all.loc[mask].copy()
            subset.drop(columns=[OUTPUT_COL], inplace=True)
            fixed = refine_with_llm(subset)
            _update_master_from_fixed(df_all, fixed)
    
            post_n = int(_needs_fix_mask(df_all).sum())
            print(f"[Pass {p}] remaining with 'See Evidence': {post_n}")
            if post_n == 0:
                break
    
        #_write_any(df_all, out_path)
        print(f"Saved cleaned file ")
        return df_all

    fixed = rerun_llm_until_clean(df_refined.copy())
    return fixed
    
