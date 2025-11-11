# extractors/__init__.py
from typing import List, Dict, Any, Optional

# -----------------------------------
# Rank-specific extractors (precise, run first)
# -----------------------------------
from .rank import (
    try_answer_rank_striking,
    try_answer_rank_nage,
    try_answer_rank_jime,
    try_answer_rank_weapons,       # optional; safe to leave even if not used
    try_answer_rank_requirements,  # whole-rank block slicer
)

# -----------------------------------
# Concept extractors
# -----------------------------------
from .kyusho import try_answer_kyusho
from .kihon_happo import try_answer_kihon_happo
from .sanshin import try_answer_sanshin

# Schools: profile handled *in app.py* (correct)
# Only the old list-extractor has been removed.

from .leadership import try_extract_answer as try_leadership

# Weapons (profiles, non-rank)
from .weapons import (
    try_answer_weapon_profile,
)

# -----------------------------------
# Master dispatcher for deterministic,
# context-only extractions.
# -----------------------------------
def try_extract_answer(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic, context-only answers for high-signal queries.
    These run BEFORE the LLM.
    Ordering matters.
    """

    # ---- Rank: striking (kicks/punches)
    ans = try_answer_rank_striking(question, passages)
    if ans:
        return ans

    # ---- Rank: throws (nage waza)
    ans = try_answer_rank_nage(question, passages)
    if ans:
        return ans

    # ---- Rank: chokes (jime waza)
    ans = try_answer_rank_jime(question, passages)
    if ans:
        return ans

    # ---- Rank: weapons (optional)
    try:
        ans = try_answer_rank_weapons(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # ---- Weapons: profile (non-rank)
    try:
        ans = try_answer_weapon_profile(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # ---- Kyusho
    try:
        ans = try_answer_kyusho(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # ---- Kihon Happo
    try:
        ans = try_answer_kihon_happo(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # ---- Sanshin
    try:
        ans = try_answer_sanshin(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # Schools note:
    #   School *profiles* run in app.py AFTER this dispatcher
    #   because they depend on UI toggles (bullets/paragraph, crisp/chatty).

    # ---- Leadership (sōke)
    try:
        ans = try_leadership(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # Nothing matched → let app.py call school-profiles or fall back to LLM
    return None
