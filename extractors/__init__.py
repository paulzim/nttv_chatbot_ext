from typing import List, Dict, Any, Optional

# -------------------------------
# Rank-specific extractors (run first after technique)
# -------------------------------
from .rank import (
    try_answer_rank_requirements,  # whole-rank block slicer (e.g., "requirements for 3rd kyu")
    try_answer_rank_striking,      # kicks/punches for a given kyu
    try_answer_rank_nage,          # throws for a given kyu
    try_answer_rank_jime,          # chokes for a given kyu
    try_answer_rank_weapons,       # optional: weapons per-rank if implemented
)

# -------------------------------
# Weapons extractors (non-rank & rank-intro)
# -------------------------------
# Some repos may not have both; import defensively.
try:
    from .weapons import (
        try_answer_weapon_profile,  # e.g., "what is a kusari fundo?"
        try_answer_weapon_rank,     # e.g., "at what rank do I learn kusari fundo?"
    )
    _HAS_WEAPON_RANK = True
except Exception:
    try:
        from .weapons import try_answer_weapon_profile
        _HAS_WEAPON_RANK = False
    except Exception:
        # Weapons not present
        try_answer_weapon_profile = None  # type: ignore
        _HAS_WEAPON_RANK = False

# -------------------------------
# Technique extractor (deterministic definitions)
# -------------------------------
try:
    from .techniques import try_answer_technique  # e.g., "what is omote gyaku"
    _HAS_TECHNIQUES = True
except Exception:
    try_answer_technique = None  # type: ignore
    _HAS_TECHNIQUES = False

# -------------------------------
# Concept extractors (context-only)
# -------------------------------
from .kyusho import try_answer_kyusho
from .kihon_happo import try_answer_kihon_happo
from .sanshin import try_answer_sanshin

# NOTE: School list/profile logic is routed in app.py
# (We call the dedicated schools extractors from there to honor UI toggles.)

# -------------------------------
# Leadership (sōke/lineage)
# -------------------------------
from .leadership import try_extract_answer as try_leadership


# ===============================
# Master dispatcher (deterministic, context-only)
# ===============================
def try_extract_answer(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic, context-only answers for high-signal queries.
    Returns a short string/bulleted block or None to fall back to the LLM.
    Ordering matters.
    """

    # ---- Technique definitions (e.g., "what is omote gyaku")
    if _HAS_TECHNIQUES and try_answer_technique:
        try:
            ans = try_answer_technique(question, passages)
            if ans:
                return ans
        except Exception:
            pass

    # ---- Rank: whole-block requirements (most specific)
    ans = try_answer_rank_requirements(question, passages)
    if ans:
        return ans

    # ---- Rank: striking / throws / chokes
    ans = try_answer_rank_striking(question, passages)
    if ans:
        return ans

    ans = try_answer_rank_nage(question, passages)
    if ans:
        return ans

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

    # ---- Weapons: rank-intro (optional) then profile
    if _HAS_WEAPON_RANK:
        try:
            ans = try_answer_weapon_rank(question, passages)  # e.g., "at what rank do I learn X?"
            if ans:
                return ans
        except Exception:
            pass

    if try_answer_weapon_profile:
        try:
            ans = try_answer_weapon_profile(question, passages)  # e.g., "what is X?"
            if ans:
                return ans
        except Exception:
            pass

    # ---- Concepts: kyusho / kihon happo / sanshin
    try:
        ans = try_answer_kyusho(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    try:
        ans = try_answer_kihon_happo(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    try:
        ans = try_answer_sanshin(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # ---- Leadership (soke & current heads)
    try:
        ans = try_leadership(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # No deterministic hit → let app.py route schools or fall back to LLM
    return None
