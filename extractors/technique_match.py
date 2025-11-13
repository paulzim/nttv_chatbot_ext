# extractors/technique_match.py
from __future__ import annotations
from typing import Tuple, Optional, List
import re
import unicodedata

from .technique_aliases import TECH_ALIASES, expand_with_aliases

def fold(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9\s\-']", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s

def technique_name_variants(name: str) -> List[str]:
    """Generate simple matching variants (strip 'no kata', handle hyphens, spacing)."""
    base = fold(name)
    variants = {base}
    # strip ' no kata'
    variants.add(re.sub(r"\bno kata\b", "", base).strip())
    # collapse spaces/hyphens
    variants.add(base.replace(" - ", " ").replace("-", " "))
    return [v for v in variants if v]

def is_single_technique_query(q: str) -> bool:
    qf = fold(q)
    # intent words + at least one alias/canonical appears
    intent = any(w in qf for w in [
        "what is", "explain", "describe", "define", "show me", "tell me about"
    ])
    if not intent:
        return False
    # see if a known technique name (canon or alias) is present
    for canon, aliases in TECH_ALIASES.items():
        for tok in technique_name_variants(canon) + [fold(a) for a in aliases]:
            if tok and tok in qf:
                return True
    return False

def canonical_from_query(q: str) -> Optional[str]:
    """Return the canonical technique name if the query mentions one."""
    qf = fold(q)
    # exact/canonical detection first
    for canon in TECH_ALIASES:
        for v in technique_name_variants(canon):
            if v and v in qf:
                return canon
    # alias expansion
    al = expand_with_aliases(q)
    if al:
        # map first alias hit back to its canon
        first = al[0]
        for canon, aliases in TECH_ALIASES.items():
            if first == fold(canon) or first in [fold(a) for a in aliases]:
                return canon
    return None
