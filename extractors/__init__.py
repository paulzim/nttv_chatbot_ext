# extractors/__init__.py
from typing import List, Dict, Any, Optional

# ----- Rank-specific extractors (most precise; run first)
from .rank import (
    try_answer_rank_striking,
    try_answer_rank_nage,
    try_answer_rank_jime,
    try_answer_rank_requirements,   # explicit “requirements for X kyu”
)

# If you implemented rank weapons in rank.py, import it; otherwise comment it out.
try:
    from .rank import try_answer_rank_weapons  # optional
except Exception:
    def try_answer_rank_weapons(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
        return None

# ----- Deterministic concept/technique extractors
from .kyusho import try_answer_kyusho
from .techniques import try_answer_technique
from .kihon_happo import try_answer_kihon_happo
from .sanshin import try_answer_sanshin

# NOTE: Schools are handled *directly in app.py*:
#   - try_answer_school_profile
#   - try_answer_schools_list
#   - is_school_list_query
# So we do not import any school extractor here to avoid symbol drift.

# Leadership (sokeship mapping, discrete facts)
from .leadership import try_extract_answer as try_leadership


def try_extract_answer(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic, context-only answers for high-signal intents.
    Return a short string (bullets or brief paragraph) or None to fall back to the LLM.
    Order matters: most specific first.
    """

    # --- Rank-specific: Striking (kicks/punches)
    ans = try_answer_rank_striking(question, passages)
    if ans:
        return ans

    # --- Rank-specific: Throws (Nage waza)
    ans = try_answer_rank_nage(question, passages)
    if ans:
        return ans

    # --- Rank-specific: Chokes (Jime waza)
    ans = try_answer_rank_jime(question, passages)
    if ans:
        return ans

    # --- Rank-specific: Requirements (ENTIRE block for an explicit “requirements for X kyu” q)
    ans = try_answer_rank_requirements(question, passages)
    if ans:
        return ans

    # --- Rank-specific: Weapons by rank (optional)
    try:
        ans = try_answer_rank_weapons(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # --- Concept: Kyusho (short, deterministic)
    try:
        ans = try_answer_kyusho(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # --- Techniques (e.g., Omote Gyaku, Musha Dori, Jumonji no Kata, etc.)
    try:
        ans = try_answer_technique(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # --- Concept: Kihon Happo
    try:
        ans = try_answer_kihon_happo(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # --- Concept: Sanshin
    try:
        ans = try_answer_sanshin(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # --- Leadership (Soke / headmaster lookups)
    try:
        ans = try_leadership(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # Schools are handled in app.py BEFORE this dispatcher.
    return None
