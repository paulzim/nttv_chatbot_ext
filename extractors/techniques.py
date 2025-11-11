# extractors/techniques.py
import re
from typing import List, Dict, Any, Optional

# Simple alias map so “what is omote gyaku”, “omote-gyaku”, etc. normalize
ALIASES = {
    "omote gyaku": ["omote-gyaku", "omote gyaku ken sabaki", "omote gyaku kensabaki"],
    "ura gyaku": ["ura-gyaku"],
    "musha dori": ["musha-dori", "musha dori"],
    "take ori": ["take-ori", "takeori"],
    "hon gyaku jime": ["hon-gyaku jime", "hon gyaku-jime"],
    "oni kudaki": ["oni-kudaki", "ude garame (oni kudaki)", "keylock - oni kudaki"],
    "ude garame": ["kimura", "ude-garame"],
    "ganseki otoshi": ["gan seki otoshi", "ganseki-otoshi"],
    # Add more as needed
}

# Regexes to extract “Name — short definition” lines from Technique Descriptions.
# We’ll look for patterns like:
#   Omote gyaku, outer wrist lock
#   Ura gyaku - inner wrist lock
#   Omote Gyaku Ken Sabaki, twisting wrist lock from a punch
LINE_PATTERNS = [
    r"^\s*([A-Z][A-Za-z0-9 \-\u00C0-\u017F']+?)\s*[,–-]\s*([^\n\r]+?)\s*$",
]

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def _is_technique_q(q: str) -> Optional[str]:
    ql = _norm(q)
    # trivial heuristic: questions like "what is X", "define X", "explain X"
    # then extract the trailing term
    m = re.search(r"(what\s+is|define|explain)\s+(.+)$", ql)
    candidate = (m.group(2) if m else ql)
    # scrub question marks and boilerplate
    candidate = candidate.replace("?", " ").strip()

    # try to normalize via alias table
    for canon, alts in ALIASES.items():
        if canon in candidate:
            return canon
        for a in alts:
            if a in candidate:
                return canon
    # fallback: return the phrase itself if short and plausible
    if 2 <= len(candidate.split()) <= 4:
        return candidate
    return None

def _harvest_tech_lines(passages: List[Dict[str, Any]]) -> List[str]:
    buf = []
    for p in passages:
        text = p.get("text", "")
        if not text:
            continue
        buf.append(text)
    all_text = "\n".join(buf)
    return all_text.splitlines()

def _build_lookup_from_lines(lines: List[str]) -> Dict[str, str]:
    out = {}
    for ln in lines:
        for pat in LINE_PATTERNS:
            m = re.match(pat, ln, flags=re.IGNORECASE)
            if not m:
                continue
            name = _norm(m.group(1))
            desc = m.group(2).strip()
            if name and desc:
                out[name] = desc
    return out

def try_answer_technique(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic definition lookup for “what is <technique>” type questions.
    Searches Technique Descriptions style lines and returns a short, precise definition.
    """
    term = _is_technique_q(question)
    if not term:
        return None

    # Build a normalized lookup table from the retrieved passages
    lines = _harvest_tech_lines(passages)
    table = _build_lookup_from_lines(lines)

    # direct hit
    if term in table:
        return f"{_title(term)} — {table[term].rstrip('.')}."
    # alias hit
    for canon, alts in ALIASES.items():
        if term == canon and canon in table:
            return f"{_title(canon)} — {table[canon].rstrip('.')}."
        if term in alts and canon in table:
            return f"{_title(canon)} — {table[canon].rstrip('.')}."

    # fuzzy fallback: try stripping hyphens or minor punctuation
    t_simple = term.replace("-", " ")
    if t_simple in table:
        return f"{_title(t_simple)} — {table[t_simple].rstrip('.')}."

    return None

def _title(s: str) -> str:
    # Keep common ninjutsu casing rules readable
    return " ".join(w.capitalize() for w in s.split())
