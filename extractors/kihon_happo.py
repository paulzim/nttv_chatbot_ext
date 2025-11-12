# extractors/kihon_happo.py
import re
from typing import List, Dict, Any, Optional

CANON_DEF = "Kihon Happo consists of Kosshi Kihon Sanpo and Torite Goho."
CANON_KOSSHI = ["Ichimonji no Kata", "Hicho no Kata", "Jumonji no Kata"]
CANON_TORITE = ["Omote Gyaku", "Omote Gyaku Ken Sabaki", "Ura Gyaku", "Musha Dori", "Ganseki Nage"]

# Lines we should ignore if they creep into parsing
UNWANTED_HINTS = (
    "drill the kihon happo",
    "practice the kihon happo",
    "use it against attackers",
    "from all kamae",
    "#",                       # markdown headings / hash noise
    "torite goho gata",        # label line, not an item
    "the five forms of grappling",
    "kihon happo.",            # trailing hash-tagged variants
)

def _is_junk(s: str) -> bool:
    ls = s.lower().strip()
    return any(h in ls for h in UNWANTED_HINTS)

def _clean_item(s: str) -> str:
    s = s.strip(" -•\t.,;").replace("  ", " ")
    # normalize spacing/casing of known items
    s = s.replace("no  kata", "no Kata")
    return s

def _split_items(tail: str) -> List[str]:
    # split on commas/semicolons; keep short/normal items; drop junk
    parts = [p for p in re.split(r"[;,]", tail) if p.strip()]
    items = []
    for p in parts:
        p2 = _clean_item(p)
        if 2 <= len(p2) <= 60 and not _is_junk(p2):
            items.append(p2)
    return items

def _extract_lists_from_text(text: str) -> (List[str], List[str]):
    kosshi, torite = [], []
    for raw in (text or "").splitlines():
        ln = raw.strip()
        if not ln or _is_junk(ln):
            continue
        low = ln.lower()

        # Kosshi Kihon Sanpo line
        if "kosshi" in low and "sanpo" in low:
            tail = ln.split(":", 1)[1].strip() if ":" in ln else ln
            kosshi.extend(_split_items(tail))
            continue

        # Torite Goho line (goho/gohō)
        if "torite" in low and ("goho" in low or "gohō" in low):
            tail = ln.split(":", 1)[1].strip() if ":" in ln else ln
            torite.extend(_split_items(tail))
            continue

    # dedupe in order
    def dedupe(seq: List[str]) -> List[str]:
        seen = set(); out = []
        for x in seq:
            if x and x not in seen:
                out.append(x); seen.add(x)
        return out

    kosshi = dedupe(kosshi)
    torite = dedupe(torite)

    # sanity: if what we captured looks wrong (empty or full of junk), fallback to canon
    def looks_bad(items: List[str], expected: List[str]) -> bool:
        if not items:
            return True
        # if many items contain generic words or labels, treat as bad
        bad_hits = sum(1 for it in items if _is_junk(it.lower()))
        # if low overlap with canonical, also treat as bad
        overlap = sum(1 for it in items if it in expected)
        return (bad_hits > 0) or (overlap < 1)

    if looks_bad(kosshi, CANON_KOSSHI):
        kosshi = CANON_KOSSHI[:]
    else:
        # Keep only the first three, and in canonical order if possible
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

def try_answer_kihon_happo(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    ql = (question or "").lower()
    if "kihon happo" not in ql and "kihon happō" not in ql:
        return None

    # Parse lists from retrieved text; the app injects a synthetic Kihon block, but this
    # extractor is robust if that block is missing or noisy.
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

    # Deterministic output (works in both Bullets/Paragraph modes via _render_det)
    lines = ["Kihon Happo:", f"- {CANON_DEF}"]
    lines.append(f"- Kosshi Kihon Sanpo: {', '.join(kosshi)}.")
    lines.append(f"- Torite Goho: {', '.join(torite)}.")
    return "\n".join(lines)
