# extractors/technique_match.py
from typing import List, Optional
import unicodedata
import re
from .technique_aliases import expand_with_aliases

def fold(s: str) -> str:
    """Lowercase + strip diacritics for robust matching."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()

def is_single_technique_query(q: str) -> Optional[str]:
    """
    Return a candidate technique name if the query looks like a single-technique ask,
    else None. Handles 'explain/define/what is ... (no kata)?'
    Skips high-level concept queries (kihon happo, sanshin, schools, ryu).
    """
    ql = (q or "").strip().lower()
    for ban in ("kihon happo", "kihon happō", "sanshin", "school", "schools", "ryu", "ryū"):
        if ban in ql:
            return None

    m = re.search(r"(?:what\s+is|define|explain)\s+(.+)$", q, flags=re.I)
    cand = (m.group(1) if m else q).strip().rstrip("?!.")
    cand = re.sub(r"\b(technique|in ninjutsu|in bujinkan)\b", "", cand, flags=re.I).strip()
    return cand if 2 <= len(cand) <= 80 else None

def technique_name_variants(name: str) -> List[str]:
    """
    Build a robust set of match variants for a technique name using:
    - alias expansion (short form ↔ '... no Kata', spelling variants)
    - hyphen/space variants
    - diacritic-folded forms
    """
    seeds = expand_with_aliases(name)
    out = []
    seen = set()

    def push(v: str):
        if not v:
            return
        if v not in seen:
            out.append(v); seen.add(v)
        f = fold(v)
        if f and f not in seen:
            out.append(f); seen.add(f)

    for s in seeds:
        push(s)
        if "-" in s:
            push(s.replace("-", " "))
        if " " in s:
            push(s.replace(" ", "-"))

    return out
