# extractors/rank.py
import re
from typing import List, Dict, Any
from .common import dedupe_preserve

# Recognize "8th kyu", "1st kyu", etc.
_ORD_RE = re.compile(r"\b(\d{1,2})(st|nd|rd|th)\s+kyu\b", re.I)

def _parse_rank_label(q: str) -> str | None:
    m = _ORD_RE.search(q)
    if not m:
        return None
    num, suf = m.group(1), m.group(2).lower()
    return f"{num}{suf} Kyu"

def _find_rank_positions(text: str, rank_label: str) -> list[int]:
    """Find where the rank appears (handles '8th Kyu' and 'Kyu: 8th Kyu')."""
    if not text:
        return []
    pat = re.compile(
        rf"(?i)(?:\b{re.escape(rank_label)}\b|\bkyu\s*:\s*{re.escape(rank_label)}\b)"
    )
    return [m.start() for m in pat.finditer(text)]

# Accept "Striking:", "Striking –", "Striking Techniques:", etc.
# Up to 3 extra words allowed between "Striking" and separator
_STRIKING_HEAD = r"striking(?:\s+\w+){0,3}\s*(?::|[-–—])\s*"

def _extract_striking_after(text: str, start_idx: int) -> str | None:
    """From a rank occurrence, find next Striking header and capture its list."""
    window = text[start_idx:start_idx + 8000]
    m = re.search(rf"(?is){_STRIKING_HEAD}(.*)", window)
    if not m:
        return None
    seg = m.group(1)
    stop = re.search(
        r"(?im)^\s*\w[^:\n]{0,40}\s*(?::|[-–—])\s*|^\s*$|^\s*(?:kyu\s*:|\d{1,2}(?:st|nd|rd|th)\s+kyu)\b",
        seg,
    )
    return seg[:stop.start()].strip() if stop else seg.strip()

def _extract_striking_text_fallback(blob: str, rank_label: str) -> str | None:
    """Fallback: look for 'rank ... Striking...' proximity with tolerant header."""
    pat = re.compile(
        rf"(?is)(?:{re.escape(rank_label)}|kyu\s*:\s*{re.escape(rank_label)})"
        rf".{{0,2500}}?{_STRIKING_HEAD}(.*?)(?=\n\s*\w[^:\n]{{0,40}}\s*(?::|[-–—])\s*|^\s*$|kyu\s*:|\d{{1,2}}(?:st|nd|rd|th)\s+kyu\b)",
        re.M,
    )
    m = pat.search(blob)
    return m.group(1).strip() if m else None

def _split_items(s: str) -> List[str]:
    """
    Split a list tolerant to commas, semicolons, slashes, newlines, bullets (•, ・), pipes, and 'and'.
    Strip parentheses and squash spaces.
    """
    if not s:
        return []
    s = re.sub(r"\band\b", ",", s, flags=re.I)
    s = s.replace("•", ",").replace("・", ",").replace("|", ",")
    parts = re.split(r"[,\n;\/]+", s)
    items = []
    for part in parts:
        t = part.strip()
        if not t:
            continue
        t = re.sub(r"^\((.*)\)$", r"\1", t).strip()
        t = re.sub(r"\s+", " ", t)
        items.append(t)
    return dedupe_preserve(items)

def _is_kick(x: str) -> bool:
    return "geri" in x.lower()

def _is_punch(x: str) -> bool:
    lx = x.lower()
    return ("tsuki" in lx) or lx.endswith(" ken") or (" ken " in lx) or ("uraken" in lx)

# Plural-friendly intent
_KICK_INTENT  = re.compile(r"\b(kick|kicks|geri|geris)\b", re.I)
_PUNCH_INTENT = re.compile(r"\b(punch|punches|tsuki|tsukis)\b|\b(?:^|[\s\-])ken\b", re.I)
_STRIKE_INTENT= re.compile(r"\b(strike|strikes|striking)\b", re.I)

def _pick_rank_passages(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rp = [p for p in passages if "rank requirements" in (p.get("source") or "").lower()]
    return rp if rp else passages

def _find_striking_text(passages: List[Dict[str, Any]], rank_label: str) -> str | None:
    """Robust multi-stage search for the Striking section of a given rank."""
    # 1) Combined blob
    blob = "\n\n".join(p["text"] for p in passages)
    positions = _find_rank_positions(blob, rank_label)
    for pos in positions:
        st = _extract_striking_after(blob, pos)
        if st:
            return st
    st = _extract_striking_text_fallback(blob, rank_label)
    if st:
        return st
    # 2) Per-passage
    for p in passages:
        txt = p["text"]
        positions = _find_rank_positions(txt, rank_label)
        for pos in positions:
            st = _extract_striking_after(txt, pos)
            if st:
                return st
        st = _extract_striking_text_fallback(txt, rank_label)
        if st:
            return st
    return None

def try_answer_rank_striking(question: str, passages: List[Dict[str, Any]]) -> str | None:
    """
    Answers:
      - 'What are the kicks in 8th kyu?'
      - 'What are the punches in 8th kyu?'
      - 'What is the striking in 8th kyu?' (returns both)
    """
    ql = question.lower()
    rank_label = _parse_rank_label(ql)
    if not rank_label:
        return None

    wants_kicks   = bool(_KICK_INTENT.search(ql))
    wants_punches = bool(_PUNCH_INTENT.search(ql))
    wants_strikes = bool(_STRIKE_INTENT.search(ql)) or (not wants_kicks and not wants_punches)

    rank_passages = _pick_rank_passages(passages)
    striking_text = _find_striking_text(rank_passages, rank_label)
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

    # If they asked for only one category and we found none, answer gracefully
    if wants_kicks and not kicks:
        return f"{rank_label} kicks: (none listed)."
    if wants_punches and not punches:
        return f"{rank_label} punches: (none listed)."
    return None
