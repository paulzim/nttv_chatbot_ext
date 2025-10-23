# extractors/rank.py
import re
from typing import List, Dict, Any
from .common import dedupe_preserve

# Recognize rank like "8th kyu", "1st kyu", etc.
ORD_RE = re.compile(r"\b(\d{1,2})(st|nd|rd|th)\s+kyu\b", re.I)

def _wanted_rank_from_question(q: str) -> str | None:
    m = ORD_RE.search(q)
    if m:
        return f"{m.group(1)}{m.group(2).lower()} kyu"
    return None

def _extract_rank_block(text: str, rank_label: str) -> str | None:
    """
    Try to isolate the section of the rank requirements that belongs to the given rank.
    We look for the rank label and capture until the next blank line preceding another 'Kyu' or a new section.
    """
    # Accept lines like "8th Kyu", "Kyu: 8th Kyu", etc.
    rank_pat = rf"(?im)^(?:\s*(?:kyu\s*:\s*)?{re.escape(rank_label)}\s*$)"
    m = re.search(rank_pat, text)
    if not m:
        return None
    start = m.end()
    # Grab a window after the rank header; stop before the next rank header or end
    seg = text[start:start + 4000]
    stop = re.search(r"(?im)^\s*(?:\d{1,2}(?:st|nd|rd|th)\s+kyu|kyu\s*:)", seg)
    return seg[:stop.start()] if stop else seg

def _extract_striking_line(block: str) -> str | None:
    """
    Find the 'Striking:' line (or paragraph). Capture text until next section header 'Word:' on a new line.
    """
    if not block:
        return None
    m = re.search(r"(?is)^\s*striking\s*:\s*(.*?)(?=\n\s*\w[^:\n]{0,40}:\s|$)", block, flags=re.M)
    if not m:
        # fallback: single-line variant
        m = re.search(r"(?im)^\s*striking\s*:\s*(.+)$", block)
        if not m:
            return None
    return m.group(1).strip()

def _split_items(s: str) -> List[str]:
    """
    Split a comma/semicolon/line/“and” separated list into items; strip parentheses.
    """
    if not s:
        return []
    # normalize separators
    s = re.sub(r"\band\b", ",", s, flags=re.I)
    raw = re.split(r"[,\n;]+", s)
    items = []
    for r in raw:
        t = r.strip()
        if not t:
            continue
        # drop outer parentheses
        t = re.sub(r"^\((.*)\)$", r"\1", t).strip()
        # collapse inner spaces
        t = re.sub(r"\s+", " ", t)
        items.append(t)
    return dedupe_preserve(items)

def _is_kick(x: str) -> bool:
    lx = x.lower()
    return "geri" in lx

def _is_punch(x: str) -> bool:
    """
    Treat tsuki, ...-ken (fist forms), uraken as punches.
    (We *exclude* pure 'shuto' by default since it's a knife-hand strike, not a punch.)
    Adjust here later if you want to include/exclude more hand strikes.
    """
    lx = x.lower()
    return (
        "tsuki" in lx
        or lx.endswith(" ken")
        or " ken " in lx
        or "uraken" in lx
    )

def try_answer_rank_striking(question: str, passages: List[Dict[str, Any]]) -> str | None:
    """
    For questions like:
      - 'kicks in 8th kyu'
      - 'punches in 8th kyu'
      - 'striking in 8th kyu'
    Build a precise answer from the Rank Requirements file.
    """
    ql = question.lower()
    want_rank = _wanted_rank_from_question(ql)
    if not want_rank:
        return None

    wants_kicks   = bool(re.search(r"\bkick|geri\b", ql))
    wants_punches = bool(re.search(r"\bpunch|tsuki|(?:^|[\s-])ken\b", ql))
    wants_strikes = bool(re.search(r"\bstrike|striking\b", ql)) or (not wants_kicks and not wants_punches)

    # Prefer passages from the rank requirements file
    rank_passages = []
    for p in passages:
        src = (p.get("source") or "").lower()
        if "rank requirements" in src or "nttv rank requirements" in src:
            rank_passages.append(p)
    if not rank_passages:
        rank_passages = passages  # fallback

    blob = "\n\n".join(p["text"] for p in rank_passages)

    block = _extract_rank_block(blob, want_rank)
    if not block:
        return None

    striking_text = _extract_striking_line(block)
    if not striking_text:
        return None

    items = _split_items(striking_text)
    if not items:
        return None

    kicks   = [x for x in items if _is_kick(x)]
    punches = [x for x in items if _is_punch(x)]

    # Build answer according to the ask
    parts = []
    if wants_kicks or (wants_strikes and kicks):
        parts.append(f"{want_rank.title()} kicks: " + ", ".join(kicks) + "." if kicks else f"{want_rank.title()} kicks: (none listed).")
    if wants_punches or (wants_strikes and punches):
        parts.append(f"{want_rank.title()} punches: " + ", ".join(punches) + "." if punches else f"{want_rank.title()} punches: (none listed).")

    if parts:
        return " ".join(parts)

    # If they explicitly asked for only one kind and we found none, still answer gracefully
    if wants_kicks and not kicks:
        return f"{want_rank.title()} kicks: (none listed)."
    if wants_punches and not punches:
        return f"{want_rank.title()} punches: (none listed)."

    return None
