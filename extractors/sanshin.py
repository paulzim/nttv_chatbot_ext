# extractors/sanshin.py
import re
from typing import List, Dict, Any
from .common import join_oxford, dedupe_preserve

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

def try_answer_sanshin(passages: List[Dict[str, Any]]) -> str | None:
    """
    Deterministic Sanshin answer that survives inconsistent formatting:
    1) Anchor match for 'Sanshin no Kata' / 'San Shin no Kata'
    2) Parse bullets or inline lists
    3) Fallback: scan blob for the 5 forms and output in canonical order
    """
    blob = "\n\n".join(p["text"] for p in passages[:6])
    blob_low = blob.lower()
    if not (("sanshin" in blob_low) or ("san shin" in blob_low)):
        return None

    anchor = r"(?:sanshin\s+no\s+kata|san\s+shin\s+no\s+kata|sanshin\b|san\s+shin\b)"

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

    ordered = []
    if items:
        for key_low, canon in wanted:
            match = next((it for it in items if contains(key_low, it)), None)
            if match:
                ordered.append(match)
        if not ordered:
            items = []

    if not items:
        for key_low, canon in wanted:
            pat = re.compile(rf"{key_low}", flags=re.I)
            if pat.search(blob):
                ordered.append(canon)

    unique_ordered = dedupe_preserve(ordered)
    if len(unique_ordered) >= 3:
        forms = join_oxford(unique_ordered)
        return f"Sanshin no Kata consists of {forms}."
    return None
