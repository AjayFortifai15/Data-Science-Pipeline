# === ROCK-SOLID LLM REFINEMENT (Ollama gpt-oss:3b) ===========================
# Input : df["llm_explanation"]
# Output: df["llm_refined_explanation"]
import sys, os

# Ensure stdout uses UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    os.environ["PYTHONIOENCODING"] = "utf-8"

import re, time, json, requests, pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from tqdm import tqdm

def ai_explanation_part_1(df_updated_final):
    # ---------------- CONFIG ----------------
    ID_COL        = "base_id_src_po"
    INPUT_COL     = "llm_explanation"
    FINAL_INPUT_EVIDENCE = "updated_evidence_text"
    VARIANCE_COL  = "variance_summary"
    OUTPUT_COL    = "llm_refined_explanation"

    # Ollama chat endpoint + model (switched to Llama 3.2 — 3B)
    OLLAMA_URL    = "http://127.0.0.1:11434/api/chat"
    MODEL         = "llama3.2:3b"

    # Generation knobs tuned for 3B (fits 6GB VRAM comfortably)
    OPTIONS = {
        "temperature": 0.2,
        "top_p": 1,
        "num_ctx": 8192,      # Llama 3.2 smalls support 8k
        "num_predict": 1000,  # slightly lower than 70B default for speed/stability
        # You can also add: "repeat_penalty": 1.05
    }

    BANNED_TERMS = [
        r"\bfraud\b", r"\bfraudulent\b", r"\bscam\b", r"\bscamming\b",
        r"\bcorruption\b", r"\bbribe\b", r"\bbribery\b"
    ]
    HEADER_LINE = "_" * 152
    CHECKPOINT_EVERY = 0
    CHECKPOINT_PATH  = "with_llm_refined_explanations_checkpoint_06_09.parquet"

    # ---------------- SMALL PARSERS ----------------
    def _find_block(src: str, title_regex: str, stop_at_next_numbered=True) -> str:
        lines = src.splitlines()
        pat = re.compile(rf"^\s*\d+\.\s*{title_regex}\s*:?\s*$", re.I)
        start = None
        for i, ln in enumerate(lines):
            if pat.match(ln.strip()):
                start = i + 1
                break
        if start is None:
            return ""
        out = []
        for ln in lines[start:]:
            if stop_at_next_numbered and re.match(r"^\s*\d+\.\s", ln):
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
        blk = _find_block(src, r"Risk\s*Drivers")
        drivers = []
        for ln in blk.splitlines():
            t = ln.strip()
            if t.startswith("· ") or t.startswith("- "):
                drivers.append(t[2:].strip())
        return drivers

    def extract_business_impact_line(src: str) -> str:
        blk = _find_block(src, r"Business\s*Impact")
        for ln in blk.splitlines():
            t = ln.strip()
            if t and ("Suspicious" in t or "Line Value" in t or "Overall Impact" in t):
                return t
        for ln in blk.splitlines():
            if ln.strip():
                return ln.strip()
        return ""

    def source_has_no_risk(source_text: str) -> bool:
        return bool(re.search(r"\bNo\s*Risk\b", source_text, re.IGNORECASE))

    # ---------------- Price Variance helpers ----------------
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
            prefix = narr[:pos]
            suffix = narr[pos:]
            return prefix + block + suffix + "\n"
        return narr + block

    # ---------------- Prompt ----------------
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
For Executive Summary strictly pick up data from Key Description section only. Pick data correcponding to those value asked only.
DO NOT hallucinate. For Risk Flag pick data strictly from Context & Trigger Risk Scenario only.
HARD CONSTRAINTS:
- Avoid legal/accusatory terms (fraud, bribery, etc.).
- Preserve IDs and codes exactly if you reference them.
- If a value is not present in SOURCE, write "(not available)".
STRICT OUTPUT FORMAT (exact headers, bullets as shown, no extra sections):
**SARA AI — Explanation**

**Executive Summary**
• Purchase Order: <'PO <po> / Item <item>' from 'PO / Item & PR Ref' (ignore anything after '&') or '(not available)'>
• Vendor: <'code – name' from 'Vendor no - Name' or '(not available)'>
• Material: <'code — text' from 'Material – Text, Type' (text before comma) or '(not available)'>
• Risk Flag: <from 'Risk Scenario' (include sub-risks from risk drivers secions) or '(not available)'>
• Line Value: <copy 'Flagged Value: ...' or 'Net / Gross Value' ... else '(not available)'>
• Action: <short phrase from Context & Trigger like 'Needs Validation'/'High Risk' etc., else '(not available)'>

**Why It Was Flagged**
• <bullet 1 grounded strictly in SOURCE>
• <bullet 2 grounded strictly in SOURCE>
• <bullet 3 grounded strictly in SOURCE>
• <bullet 4 grounded strictly in SOURCE>
• <bullet 5 grounded strictly in SOURCE>
• <bullet 6 grounded strictly in SOURCE>
• <bullet 7 grounded strictly in SOURCE>
• <bullet 8 grounded strictly in SOURCE>
• <bullet 9 grounded strictly in SOURCE>

**Recommended Next Steps**
1. <specific, actionable step grounded in SOURCE>
2. <specific, actionable step grounded in SOURCE>
3. <specific, actionable step grounded in SOURCE>
(Optionally a 4th if clearly supported by SOURCE.)

{no_risk_note if no_risk else ""}

SOURCE (read-only; copy tokens; do not reformat)
— BEGIN —
{source_text}
— END —
""".strip()

    # ---------------- Ollama client ----------------
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
                     "Avoid legal/accusatory terms (fraud, corruption, etc.)."},
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

    # ---------------- Sanitizers & fallbacks ----------------
    def sanitize_banned_terms(text: str) -> str:
        out = text
        for pat in BANNED_TERMS:
            out = re.sub(pat, "risk", out, flags=re.IGNORECASE)
        return out

    SECTION_PATTERNS = [
        re.compile(r'^\s*\*{0,2}\s*SARA\s*AI\s*[—–-]?\s*Explanation\s*\*{0,2}\s*$', re.I),
        re.compile(r'^\s*\*{0,2}\s*Executive\s+Summary\s*\*{0,2}\s*$', re.I),
        re.compile(r'^\s*\*{0,2}\s*Why\s+It\s+Was\s+Flagged\s*\*{0,2}\s*$', re.I),
        re.compile(r'^\s*\*{0,2}\s*Price\s+Variance(?:\s+Value)?\s*\*{0,2}\s*$', re.I),
        re.compile(r'^\s*\*{0,2}\s*Recommended\s+Next\s+Steps\s*\*{0,2}\s*$', re.I),
    ]

    def ensure_sections_present(text: str, source_fallback: str) -> str:
        t = text.strip()
        if "**SARA AI — Explanation**" not in t:
            t = "**SARA AI — Explanation**\n\n" + t
        if "**Executive Summary**" not in t:
            t += "\n\n**Executive Summary**\nSee Evidence."
        if "**Why It Was Flagged**" not in t:
            t += "\n\n**Why It Was Flagged**\n· See Evidence"
        if "**Recommended Next Steps**" not in t:
            t += "\n\nRecommended Next Steps\n· Review Evidence with requester"
        if not t.strip():
            t = "**SARA AI — Explanation**\n\n**Executive Summary**\nSee Evidence.\n\n**Why It Was Flagged**\n· See Evidence\n\n**Recommended Next Step**\n· Review Evidence"
        return t

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
                    insert_block = (header_line + "\n") * times
                    lines.insert(i, insert_block)
                    i += 1
            i += 1
        return "".join(lines)

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

        why_bullets = []
        if drivers:
            for d in drivers[:4]:
                why_bullets.append(f"· Driver: {d}")
        if not why_bullets:
            why_bullets = ["· No specific risk drivers listed in the source."]

        next_steps = [
            "· Validate price justification and contract terms for this PO line.",
            "· Cross-check material/vendor pricing against recent comparable lines.",
            "· Confirm requester acknowledgment of any variance and approval trail.",
            "· Confirm vendor details for any lack of data or amount exceeding threshold."
        ]

        text = (
            "**SARA AI — Explanation**\n\n"
            "**Executive Summary**\n" + exec_summary + "\n\n"
            "**Why It Was Flagged**\n" + "\n".join(why_bullets) + "\n\n"
            "**Recommended Next Steps**\n" + "\n".join(next_steps)
        )
        return text

    # ---------------- DRIVER ----------------
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
            main_source = row.get(final_evidence, "")
            source = str(row.get(input_col, "") or "").strip()
            variance_text = str(row.get(variance_col, "") or "").strip()
            variance_block = build_price_variance_bullets(variance_text)

            if not source:
                outs.append("")
            else:
                prompt = build_refine_prompt(source)
                try:
                    narrative = client.chat(prompt)
                    if not narrative.strip():
                        narrative = local_narrative_fallback(source)
                except Exception:
                    narrative = local_narrative_fallback(source)

                # Insert our variance block (only if it has something meaningful)
                if 'NaN' not in variance_block:
                    final_doc = insert_price_variance_block(narrative, variance_block)
                else:
                    final_doc = narrative

                final_doc = sanitize_banned_terms(final_doc)
                final_doc = ensure_sections_present(final_doc, source)
                final_doc = add_header_line_before_sections(final_doc, times=1)

                final_text = final_doc.rstrip() + "\n" + main_source
                outs.append(final_text)

            if checkpoint_every and (i + 1) % checkpoint_every == 0:
                tmp = df2.iloc[: i + 1].copy()
                tmp[output_col] = pd.Series(outs, index=tmp.index, dtype="string")
                tmp.to_parquet(checkpoint_path, index=False)

        df2[output_col] = pd.Series(outs, index=df2.index, dtype="string").fillna("")
        return df2

    # Run on non–No Risk rows
    df = df_updated_final.copy()
    new_df = df[df['updated_risk_level'] != 'No Risk']
    df_refined = refine_with_llm(new_df)
    return df_refined