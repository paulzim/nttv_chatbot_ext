# extractors/sanshin.py
from __future__ import annotations
import re
from typing import List, Dict, Any, Optional
from .common import join_oxford, dedupe_preserve

# -------- existing helpers (unchanged) --------

def _collect_after_anchor(blob: str, anchor_regex: str, window: int = 3000) -> str:
    m = re.search(anchor_regex, blob, flags=re.I)
    if not m:
        return ""
    return blob[m.end(): m.end() + window]

def _parse_bullets_or_shortlines(seg: str) -> List[str]:
    lines, started = [], False
    for raw in seg.splitlines():
        r = raw.strip()
        if not r:
            if started:
                break
            continue
        if re.match(r"^[-·•]\s+", r):
            started = True
            r = re.sub(r"^[-·•]\s+", "", r)
            lines.append(r)
        else:
            # short “title-ish” lines
            if re.match(r'^[A-Z][A-Za-z0-9\s"’\-\(\)]+$', r) and len(r.split()) <= 8:
                started = True
                lines.append(r)
            elif started:
                break
    return dedupe_preserve(lines)

def _find_inline_list_near(blob: str, anchor_regex: str) -> List[str]:
    m = re.search(anchor_regex + r"\s*:\s*(.+?)(?:\n\n|$)", blob, flags=re.I | re.S)
    if not m:
        return []
    text = m.group(1)
    parts = [p.strip(" -•·\t") for p in re.split(r"[,\n]", text)]
    return [p for p in parts if p and len(p) <= 60]

# -------- new light intent detector --------

def _is_sanshin_query(question: str) -> bool:
    q = (question or "").lower()
    return (
        "sanshin" in q
        or "san shin" in q
        or "sanshin no kata" in q
        or "san shin no kata" in q
        or ("five" in q and "elements" in q and "kata" in q)
    )

# -------- public API (router-compatible) --------

def try_answer_sanshin(
    question: str,
    passages: List[Dict[str, Any]] | None = None,
) -> Optional[str]:
    """
    Deterministic Sanshin explainer that:
      1) Uses question intent to avoid accidental triggers,
      2) Tries to parse nearby bullets/inline lists from context,
      3) Falls back to canonical list if needed.

    Signature accepts (question, passages) to match router usage.
    """
    if not _is_sanshin_query(question):
        return None

    passages = passages or []
    blob = "\n\n".join(p.get("text", "") for p in passages[:6])
    blob_low = blob.lower()

    anchor = r"(?:sanshin\s+no\s+kata|san\s+shin\s+no\s+kata|sanshin\b|san\s+shin\b)"

    # If the corpus mentions Sanshin, try to extract ordered items
    ordered: List[str] = []
    if ("sanshin" in blob_low) or ("san shin" in blob_low):
        seg = _collect_after_anchor(blob, anchor)
        items = _parse_bullets_or_shortlines(seg)
        if not items:
            items = _find_inline_list_near(blob, anchor)

        wanted = [
            ("chi no kata", "Chi no Kata"),
            ("sui no kata", "Sui no Kata"),
            ("ka no kata",  "Ka no Kata"),
            ("fu no kata",  "Fu no Kata"),
            ("ku no kata",  "Ku no Kata"),
        ]

        def contains(name_low: str, text: str) -> bool:
            return name_low in text.lower()

        if items:
            for key_low, canon in wanted:
                match = next((it for it in items if contains(key_low, it)), None)
                if match:
                    ordered.append(match)

        # If bullets/inline not found, scan the entire blob in canonical order
        if not ordered:
            for key_low, canon in wanted:
                pat = re.compile(rf"{key_low}", flags=re.I)
                if pat.search(blob):
                    ordered.append(canon)

        ordered = dedupe_preserve(ordered)

        if len(ordered) >= 3:
            return f"Sanshin no Kata consists of {join_oxford(ordered)}."

    # Fallback: canonical, accurate answer if context was thin
    return (
        "Sanshin no Kata (Five Elements) consists of "
        "Chi no Kata, Sui no Kata, Ka no Kata, Fu no Kata, and Ku no Kata."
    )
