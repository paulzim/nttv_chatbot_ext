# extractors/rank.py
import re
from typing import List, Dict, Any
from .common import dedupe_preserve

# Recognize rank like "8th kyu", "1st kyu", etc.
_ORD_RE = re.compile(r"\b(\d{1,2})(st|nd|rd|th)\s+kyu\b", re.I)

def _parse_rank_label(q: str) -> str | None:
    """
    Extract '8th kyu' style and normalize display as '8th Kyu'
    (ordinal suffix stays lowercase, 'Kyu' capitalized).
    """
    m = _ORD_RE.search(q)
    if not m:
        return None
    num, suf = m.group(1), m.group(2).lower()
    return f"{num}{suf} Kyu"

def _find_rank_positions(text: str, rank_label: str) -> list[int]:
    """
    Return start indices where the rank appears, tolerant to formats like:
      - "8th Kyu"
      - "Kyu: 8th Kyu"
      - "Kyu : 8th Kyu | Section: Striking"
    """
    if not text:
        return []
    # Accept either "... 8th Kyu ..." or "Kyu: 8th Kyu"
    pat = re.compile(
        rf"(?i)(?:\b{re.escape(rank_label)}\b|\bkyu\s*:\s*{re.escape(rank_label)}\b)"
    )
    return [m.start() for m in pat.finditer(text)]

def _extract_striking_after(text: str, start_idx: int) -> str | None:
    """
    From a given index (rank occurrence), find the nearest 'Striking:' section that follows.
    Capture until the next header 'Word:' on a new line, a double newline, or the next rank header.
    Handles inline cases where '... | Section: Striking: ...' appears on the same line.
    """
    window = text[start_idx:start_idx + 6000]

    # Look for 'Striking:' directly
    m = re.search(r"(?is)striking\s*:\s*(.*)", window)
    if not m:
        return None

    # Start right after 'Striking:'
    seg = m.group(1)

    # Stop at next header (e.g., 'Weapon:', 'Kamae:', 'Blocking:'), next double newline, or next 'Kyu:'/'Nth Kyu'
    stop = re.search(
        r"(?im)^\s*\w[^:\n]{0,40}:\s|^\s*$|^\s*(?:kyu\s*:|\d{1,2}(?:st|nd|rd|th)\s+kyu)\b",
        seg
    )
    if stop:
        return seg[:stop.start()].strip()
    return seg.strip()

def _extract_striking_text_fallback(blob: str, rank_label: str) -> str | None:
    """
    Fallback: search the whole blob for a small window that has both the rank and a
    nearby 'Striking:' even if they are not in strict header/section structure.
    """
    # Try a 'rank ... Striking:' pattern within ~2000 chars
    pat = re.compile(
        rf"(?is)(?:{re.escape(rank_label)}|kyu\s*:\s*{re.escape(rank_label)})"
        r".{0,2000}?striking\s*:\s*(.*?)(?=\n\s*\w[^:\n]{0,40}:\s|^\s*$|kyu\s*:|\d{1,2}(?:st|nd|rd|th)\s+kyu\b)",
        re.M
    )
    m = pat.search(blob)
    return m.group(1).strip() if m else None

def _split_items(s: str) -> List[str]:
    """
    Split a comma/semicolon/slash/line/“and” separated list into atomic items; strip outer parentheses.
    """
    if not s:
        return []
    s = re.sub(r"\band\b", ",", s, flags=re.I)
    parts = re.split(r"[,\n;\/]+", s)
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
    return "geri" in x.lower()  # e.g., yoko geri, mae geri, sokuho geri

def _is_punch(x: str) -> bool:
    lx = x.lower()
    return (
        "tsuki" in lx
        or lx.endswith(" ken")
        or " ken " in lx
        or "uraken" in lx
    )

# Plural-friendly intent regexes
_KICK_INTENT = re.compile(r"\b(kick|kicks|geri|geris)\b", re.I)
_PUNCH_INTENT = re.compile(r"\b(punch|punches|tsuki|tsukis)\b|\b(?:^|[\s\-])ken\b", re.I)
_STRIKE_INTENT = re.compile(r"\b(strike|strikes|striking)\b", re.I)

def _pick_rank_passages(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rp = [p for p in passages if "rank requirements" in (p.get("source") or "").lower()]
    return rp if rp else passages

def try_answer_rank_striking(question: str, passages: List[Dict[str, Any]]) -> str | None:
    """
    Answers:
      - 'What are the kicks in 8th kyu?' / 'What kicks do I need to know for 8th kyu?'
      - 'What are the punches in 8th kyu?'
      - 'What is the striking in 8th kyu?' (returns both buckets)
    """
    ql = question.lower()
    rank_label = _parse_rank_label(ql)
    if not rank_label:
        return None

    wants_kicks   = bool(_KICK_INTENT.search(ql))
    wants_punches = bool(_PUNCH_INTENT.search(ql))
    # If neither kicks nor punches explicitly asked but 'striking' is, return both.
    wants_strikes = bool(_STRIKE_INTENT.search(ql)) or (not wants_kicks and not wants_punches)

    # Prefer the rank requirements file
    rank_passages = _pick_rank_passages(passages)

    # Try to find 'Striking:' near a rank mention (robust to inline formatting)
    blob = "\n\n".join(p["text"] for p in rank_passages)
    positions = _find_rank_positions(blob, rank_label)
    striking_text = None

    for pos in positions:
        striking_text = _extract_striking_after(blob, pos)
        if striking_text:
            break

    if not striking_text:
        # Fallback: global scan for 'rank ... Striking: ...'
        striking_text = _extract_striking_text_fallback(blob, rank_label)

    if not striking_text:
        # Last-chance: scan each passage separately (handles cases where chunking splits)
        for p in rank_passages:
            positions = _find_rank_positions(p["text"], rank_label)
            for pos in positions:
                striking_text = _extract_striking_after(p["text"], pos)
                if striking_text:
                    break
            if not striking_text:
                striking_text = _extract_striking_text_fallback(p["text"], rank_label)
            if striking_text:
                break

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
        parts.append(f"{rank_label} punches: " + (", ".join(punches) if punches else "(none listed)") + ".")

    if parts:
        return " ".join(parts)

    # Graceful fallback if they asked specifically for one kind but found none
    if wants_kicks and not kicks:
        return f"{rank_label} kicks: (none listed)."
    if wants_punches and not punches:
        return f"{rank_label} punches: (none listed)."
    return None
