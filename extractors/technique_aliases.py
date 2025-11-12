# extractors/technique_aliases.py
from typing import Dict, List

# Canonical technique/kata names → common aliases (short forms, alt spellings)
KATA_ALIASES: Dict[str, List[str]] = {
    # Kihon Happo (Kosshi)
    "Ichimonji no Kata": ["Ichimonji"],
    "Hicho no Kata": ["Hicho", "Hichō"],
    "Jumonji no Kata": ["Jumonji", "Jūmonji"],

    # Sanshin (often asked by short name)
    "Chi no Kata": ["Chi"],
    "Sui no Kata": ["Sui"],
    "Ka no Kata": ["Ka"],
    "Fu no Kata": ["Fu"],
    "Ku no Kata": ["Ku"],

    # Common joint-locks & throws — include hyphen/spacing variants
    "Omote Gyaku": ["Omote-Gyaku"],
    "Ura Gyaku": ["Ura-Gyaku"],
    "Musha Dori": ["Musha-Dori", "Musha-dori"],
    "Ganseki Otoshi": ["Ganseki-Otoshi", "Ganseki-otoshi"],
}

def expand_with_aliases(name: str) -> List[str]:
    """
    Given a candidate technique name, return a list of equivalent variants,
    including short forms for '... no Kata' and known alias spellings.
    """
    name = (name or "").strip()
    if not name:
        return []

    variants = {name}
    # If user provided a short form, generate the '... no Kata' variant too
    for canon, aliases in KATA_ALIASES.items():
        if name == canon or name in aliases:
            variants.add(canon)
            variants.update(aliases)

    # Also handle generic "... no Kata" <-> short form even if not in table
    low = name.lower()
    if low.endswith(" no kata"):
        variants.add(name[:-8].strip())  # drop " no Kata"
    else:
        variants.add(f"{name} no Kata")

    return list(dict.fromkeys(v for v in variants if v))  # de-dup preserve order
