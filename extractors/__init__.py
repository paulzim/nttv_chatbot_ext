# extractors/__init__.py
from typing import List, Dict, Any, Optional

# Rank-specific extractors (most precise; run first)
from .rank import (
    try_answer_rank_striking,
    try_answer_rank_nage,
    try_answer_rank_jime,
    # If you added weapons/buki support in rank.py, keep the next import; otherwise remove it.
    try_answer_rank_weapons,  # optional, depending on your rank.py
)

# Concept extractors (deterministic, context-only)
from .kyusho import try_answer_kyusho           # should run right after rank
from .kihon_happo import try_answer_kihon_happo
from .sanshin import try_answer_sanshin
from .schools import try_answer_schools


def try_extract_answer(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic, context-only answers for high-signal intents.
    Return a single short string or None to fall back to the LLM.
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

    # --- Rank-specific: Weapons/Buki (optional, if implemented)
    try:
        ans = try_answer_rank_weapons(question, passages)  # remove if not using weapons
        if ans:
            return ans
    except Exception:
        pass

    # --- Concepts: Kyusho (place BEFORE other concepts so it short-circuits reliably)
    try:
        ans = try_answer_kyusho(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # --- Concepts: Kihon Happo
    try:
        ans = try_answer_kihon_happo(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # --- Concepts: Sanshin
    try:
        ans = try_answer_sanshin(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # --- Concepts: Schools
    try:
        ans = try_answer_schools(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    return None
