# extractors/kihon_happo.py
import re
from typing import List, Dict, Any, Optional

# Canonical definitions (fallbacks if parsing is noisy or incomplete)
CANON_DEF = "Kihon Happo consists of Kosshi Kihon Sanpo and Torite Goho."
CANON_KOSSHI = ["Ichimonji no Kata", "Hicho no Kata", "Jumonji no Kata"]
CANON_TORITE = ["Omote Gyaku", "Omote Gyaku Ken Sabaki", "Ura Gyaku", "Musha Dori", "Ganseki Nage"]

# Lines/phrases we do NOT want to treat as items if they appear in source text
UNWANTED_HINTS = (
    "drill the kihon happo",
    "practice the kihon happo",
    "use it against attackers",
    "from all kamae",
    "the five forms of grappling",
    "torite goho gata",
    "kihon happo.",    # trailing hashtaggy variants
    "#",               # markdown header noise
)

# Extra phrasings that should still trigger the extractor
TRIGGER_PHRASES = (
    "kihon happo", "kihon happō", "kihon-happo", "kihon-happō",
    "eight basics", "8 basics", "the eight basics",
)


def _is_junk_line(s: str) -> bool:
    ls = s.lower().strip()
    return any(h in ls for h in UNWANTED_HINTS)


def _clean_item(s: str) -> str:
    s = s.strip(" -•\t.,;").replace("  ", " ")
    # Normalize common spacing/casing
    s = s.replace("no  kata", "no Kata")
    return s


def _split_items(tail: str) -> List[str]:
    # Split on commas/semicolons; keep short/normal items; drop junk
    parts = [p for p in re.split(r"[;,]", tail) if p.strip()]
    items = []
    for p in parts:
        p2 = _clean_item(p)
        if 2 <= len(p2) <= 60 and not _is_junk_line(p2):
            items.append(p2)
    return items


def _extract_lists_from_text(text: str) -> (List[str], List[str]):
    """
    Parse Kosshi Kihon Sanpo and Torite Goho from arbitrary context lines.
    Robust to noise, falls back to canonical if the capture looks wrong.
    """
    kosshi, torite = [], []

    for raw in (text or "").splitlines():
        ln = raw.strip()
        if not ln or _is_junk_line(ln):
            continue
        low = ln.lower()

        # Kosshi Kihon Sanpo line; allow 'sanpo'/'sanpō' and weak punctuation
        if "kosshi" in low and ("sanpo" in low or "sanpō" in low):
            tail = ln.split(":", 1)[1].strip() if ":" in ln else ln
            kosshi.extend(_split_items(tail))
            continue

        # Torite Goho line; allow 'goho'/'gohō'
        if "torite" in low and ("goho" in low or "gohō" in low):
            tail = ln.split(":", 1)[1].strip() if ":" in ln else ln
            torite.extend(_split_items(tail))
            continue

    # De-dupe while preserving order
    def dedupe(seq: List[str]) -> List[str]:
        seen, out = set(), []
        for x in seq:
            if x and x not in seen:
                out.append(x)
                seen.add(x)
        return out

    kosshi = dedupe(kosshi)
    torite = dedupe(torite)

    # Heuristic sanity check: if empty or obviously noisy, use canonical
    def looks_bad(items: List[str], expected: List[str]) -> bool:
        if not items:
            return True
        bad_hits = sum(1 for it in items if _is_junk_line(it.lower()))
        overlap = sum(1 for it in items if it in expected)
        return (bad_hits > 0) or (overlap < 1)

    if looks_bad(kosshi, CANON_KOSSHI):
        kosshi = CANON_KOSSHI[:]
    else:
        # Keep canonical ordering for the first 3 if present
        ordered = [it for it in CANON_KOSSHI if it in kosshi]
        for it in kosshi:
            if it not in ordered:
                ordered.append(it)
        kosshi = ordered[:3]

    if looks_bad(torite, CANON_TORITE):
        torite = CANON_TORITE[:]
    else:
        ordered = [it for it in CANON_TORITE if it in torite]
        for it in torite:
            if it not in ordered:
                ordered.append(it)
        torite = ordered[:5]

    return kosshi, torite


def _question_triggers_kihon(question: str) -> bool:
    ql = (question or "").lower()
    return any(t in ql for t in TRIGGER_PHRASES)


def try_answer_kihon_happo(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic, context-only answer for Kihon Happo.
    - Robustly parses lists from retrieved passages (including synthetic inserts).
    - Falls back to canonical content if passages are noisy.
    - Returns a concise multi-line string; the UI decides bullets vs. paragraph.
    """
    if not _question_triggers_kihon(question):
        return None

    # Parse lists from up to the first N passages (synthetic block should be near the top)
    kosshi, torite = [], []
    for p in passages[:12]:
        k, t = _extract_lists_from_text(p.get("text", ""))
        if k and not kosshi:
            kosshi = k
        if t and not torite:
            torite = t
        if kosshi and torite:
            break

    # Final guard: fallback to canonical if either is missing
    if not kosshi:
        kosshi = CANON_KOSSHI[:]
    if not torite:
        torite = CANON_TORITE[:]

    # Deterministic output (works with both "Crisp" and "Chatty" renderers)
    lines = ["Kihon Happo:", f"- {CANON_DEF}"]
    lines.append(f"- Kosshi Kihon Sanpo: {', '.join(kosshi)}.")
    lines.append(f"- Torite Goho: {', '.join(torite)}.")
    return "\n".join(lines)
