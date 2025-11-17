# extractors/__init__.py
from typing import List, Dict, Any, Optional

# ----- Rank-specific extractors (most precise; run first)
from .rank import (
    try_answer_rank_striking,
    try_answer_rank_nage,
    try_answer_rank_jime,
    try_answer_rank_requirements,   # explicit "requirements for X kyu"
)

# If you implemented rank weapons in rank.py, import it; otherwise provide a no-op.
try:
    from .rank import try_answer_rank_weapons  # optional
except Exception:  # pragma: no cover - fallback in case rank weapons isn't implemented
    def try_answer_rank_weapons(
        question: str, passages: List[Dict[str, Any]]
    ) -> Optional[str]:
        return None


# ----- Weapon profiles (Hanbo, Kusari Fundo, Katana, Shuriken types, etc.)
# These work off NTTV Weapons Reference.txt and related sources.
try:
    from .weapons import (
        try_answer_weapon_profile,   # weapon overview (hanbo, katana, etc.)
        try_answer_katana_parts,     # special case: parts of the katana
    )
except Exception:  # pragma: no cover
    def try_answer_weapon_profile(
        question: str, passages: List[Dict[str, Any]]
    ) -> Optional[str]:
        return None

    def try_answer_katana_parts(
        question: str, passages: List[Dict[str, Any]]
    ) -> Optional[str]:
        return None


# ----- Deterministic concept/technique extractors
from .kyusho import try_answer_kyusho
from .kihon_happo import try_answer_kihon_happo   # keep this before generic techniques
from .techniques import try_answer_technique
from .sanshin import try_answer_sanshin           # must accept (question, passages)

# Leadership (Soke lookups)
from .leadership import try_extract_answer as try_leadership

# Glossary fallback (single-term / definition questions)
try:
    from .glossary import try_answer_glossary
except Exception:  # pragma: no cover
    def try_answer_glossary(
        question: str, passages: List[Dict[str, Any]]
    ) -> Optional[str]:
        return None


def try_extract_answer(
    question: str, passages: List[Dict[str, Any]]
) -> Optional[str]:
    """
    Deterministic, context-only answers for high-signal intents.
    Return a short string or None to fall back to the LLM/generic path.
    Order matters: most specific first.
    """

    # --- Rank-specific: Striking / Throws / Chokes
    ans = try_answer_rank_striking(question, passages)
    if ans:
        return ans

    ans = try_answer_rank_nage(question, passages)
    if ans:
        return ans

    ans = try_answer_rank_jime(question, passages)
    if ans:
        return ans

    # --- Rank-specific: Requirements (ENTIRE block for "requirements for X kyu")
    ans = try_answer_rank_requirements(question, passages)
    if ans:
        return ans

    # --- Rank-specific: Weapons by rank (optional)
    ans = try_answer_rank_weapons(question, passages)
    if ans:
        return ans

    # --- Katana parts (very specific intent: parts/terminology of the katana)
    ans = try_answer_katana_parts(question, passages)
    if ans:
        return ans

    # --- Weapon profiles (Hanbo, Kusari Fundo, Katana, Shuriken, etc.)
    # This is where questions like "What is the hanbo weapon?" or
    # "What are the types of shuriken?" should land.
    ans = try_answer_weapon_profile(question, passages)
    if ans:
        return ans

    # --- Concept: Kyusho (short, deterministic)
    ans = try_answer_kyusho(question, passages)
    if ans:
        return ans

    # --- Kihon Happo (run BEFORE techniques so it wins over general technique matches)
    ans = try_answer_kihon_happo(question, passages)
    if ans:
        return ans

    # --- Techniques (Omote Gyaku, Musha Dori, Jumonji no Kata, etc.)
    ans = try_answer_technique(question, passages)
    if ans:
        return ans

    # --- Concept: Sanshin
    ans = try_answer_sanshin(question, passages)
    if ans:
        return ans

    # --- Leadership (Soke / headmaster)
    ans = try_leadership(question, passages)
    if ans:
        return ans

    # --- Glossary fallback (single-term Ninjutsu/Bujinkan terms)
    ans = try_answer_glossary(question, passages)
    if ans:
        return ans

    return None
