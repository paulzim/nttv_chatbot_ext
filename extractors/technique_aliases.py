# extractors/technique_aliases.py
from __future__ import annotations
from typing import Dict, List

# Minimal alias map; extend freely.
# Key = canonical technique name as it appears at the start of the MD line.
TECH_ALIASES: Dict[str, List[str]] = {
    # Wrist locks
    "Omote Gyaku": ["omote gyaku", "forward wrist lock", "outer wrist lock"],
    "Ura Gyaku": ["ura gyaku", "reverse wrist lock", "inside wrist lock"],

    # Throws / controls (examples)
    "Musha Dori": ["musha dori", "warrior capture"],
    "Ganseki Nage": ["ganseki nage", "rock throw"],

    # Kihon kata (examples)
    "Jumonji no Kata": ["jumonji no kata", "jumonji", "cross form"],

    # The one we’re fixing now:
    "Oni Kudaki": ["oni kudaki", "ogre crusher", "demon crusher"],

    # Add more as needed...
}

def expand_with_aliases(q: str) -> List[str]:
    """Return lowercased aliases that could match q."""
    ql = q.lower().strip()
    out: List[str] = []
    for canon, aliases in TECH_ALIASES.items():
        tokens = [canon.lower()] + [a.lower() for a in aliases]
        if any(tok in ql for tok in tokens):
            out.extend(tokens)
    # de-dup while keeping order
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq
