# extractors/rank.py
import re
from typing import List, Dict, Any
from .common import dedupe_preserve

# Recognize rank like "8th kyu", "1st kyu", etc.
_ORD_RE = re.compile(r"\b(\d{1,2})(st|nd|rd|th)\s+kyu\b", re.I)

def _parse_rank_label(q: str) -> str | None:
    """
    Extract a rank label from the user question and return it formatted as '8th Kyu'
    (keeping the ordinal suffix lowercase).
    """
    m = _ORD_RE.search(q)
    if not m:
        return None
    num, suf = m.group(1), m.group(2).lower()
    return f"{num}{suf} Kyu"   # nice formatting; use lowercase suffix, capital K in Kyu

def _extract_rank_block(text: str, rank_label: str) -> str | None:
    """
    Isolate the section of the rank requirements that belongs to the given rank.
    Accepts lines like '8th Kyu' or 'Kyu: 8th Kyu'. Captures until the next rank header.
    """
    # case-insensitive, multiline
    rank_pat = rf"(?im)^(?:\s*(?:kyu\s*:\s*)?{re.escape(rank_label)}\s*$)"
    m = re.search(rank_pat, text)
    if not m:
        return None
    start = m.end()
    seg = text[start:start + 5000]  # window after the rank header
    # Stop at next rank header or "Kyu:" style block
    stop = re.search(r"(?im)^\s*(?:\d{1,2}(?:st|nd|rd|th)\s+kyu|kyu\s*:)\b", seg)
    return seg[:stop.start()] if stop else seg

def _extract_striking_text(block: str) -> str | None:
    """
    Find the 'Striking:' section. Capture text until the next header like 'Something:' or end.
    Tolerates extra spaces and case differences.
    """
    if not block:
        return None
    m = re.search(r"(?is)^\s*striking\s*:\s*(.*?)(?=\n\s*\w[^:\n]{0,40}:\s|$)", block, flags=re.M)
    if not m:
        m = re.search(r"(?im)^\s*striking\s*:\s*(.+)$", block)
        if not m:
            return None
    return m.group(1).strip()

def _split_items(s: str) -> List[str]:
    """
    Split a comma/semicolon/line/“and” separated list into atomic items; strip outer parentheses.
    """
    if not s:
        return []
    s = re.sub(r"\band\b", ",", s, flags=re.I)
    parts = re.split(r"[,\n;]+", s)
    items: List[str] = []
    for part in parts:
        t = part.strip()
        if not t:
            continue
        t = re.sub(r"^\((.*)\)$", r"\1", t).strip()
        t = re.sub(r"\s+", " ", t)
        items.append(t)
    return dedupe_preserve(items)

def _is_kick(x: str) -> bool:
    lx = x.lower()
    return "geri" in lx  # yoko geri / mae geri / etc.

def _is_punch(x: str) -> bool:
    """
    Treat tsuki, ...-ken (fist forms), and uraken as 'punches' for Q&A purposes.
    Adjust here if you want to include/exclude other hand strikes later.
    """
    lx = x.lower()
    return (
        "tsuki" in lx
        or lx.endswith(" ken")
        or " ken " in lx
        or "uraken" in lx
    )

# Plural-friendly intent regexes
_KICK_INTENT = re.compile(r"\b(kick|kicks|geri|geris)\b", re.I)
_PUNCH_INTENT = re.compile(r"\b(punch|punches|tsuki|tsukis)\b|\b(?:ken)\b|\b(?:^|[\s\-])ken\b", re.I)
_STRIKE_INTENT = re.compile(r"\b(strike|strikes|striking)\b", re.I)

def try_answer_rank_striking(question: str, passages: List[Dict[str, Any]]) -> str | None:
    """
    For questions like:
      - 'kicks in 8th kyu' / 'What are the kicks I need to know for 8th kyu?'
      - 'punches in 8th kyu'
      - 'striking in 8th kyu'
    Build a precise answer from the Rank Requirements file.
    """
    ql = question.lower()
    rank_label = _parse_rank_label(ql)
    if not rank_label:
        return None

    wants_kicks   = bool(_KICK_INTENT.search(ql))
    wants_punches = bool(_PUNCH_INTENT.search(ql))
    # If they didn’t say “kicks” or “punches” explicitly but asked about striking, treat as both.
    wants_strikes = bool(_STRIKE_INTENT.search(ql)) or (not wants_kicks and not wants_punches)

    # Prefer passages from the rank requirements file
    rank_passages = [p for p in passages if "rank requirements" in (p.get("source") or "").lower()]
    if not rank_passages:
        rank_passages = passages  # fallback

    blob = "\n\n".join(p["text"] for p in rank_passages)

    block = _extract_rank_block(blob, rank_label)
    if not block:
        return None

    striking_text = _extract_striking_text(block)
    if not striking_text:
        return None

    items = _split_items(striking_text)
    if not items:
        return None

    kicks   = [x for x in items if _is_kick(x)]
    punches = [x for x in items if _is_punch(x)]

    parts: List[str] = []
    if wants_kicks or (wants_strikes and kicks):
        parts.append(f"{rank_label} kicks: " + (", ".join(kicks) if kicks else "(none listed)") + ".")
    if wants_punches or (wants_strikes and punches):
        parts.append(f"{rank_label} punches: " + (", .".join(punches) if punches else "(none listed)") + ".")

    if parts:
        return " ".join(parts)

    # Graceful fallback if they asked specifically for one kind but found none
    if wants_kicks and not kicks:
        return f"{rank_label} kicks: (none listed)."
    if wants_punches and not punches:
        return f"{rank_label} punches: (none listed)."
    return None
