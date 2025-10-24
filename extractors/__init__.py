# extractors/__init__.py
from typing import List, Dict, Any, Optional

# Specific extractors
from .rank import (
    try_answer_rank_striking,
    try_answer_rank_nage,
    try_answer_rank_jime,
)
from .kyusho import try_answer_kyusho
from .kihon_happo import try_answer_kihon_happo  # if you added it earlier; else you can remove this line
from .sanshin import try_answer_sanshin     # if you added it earlier; else you can remove this line
from .schools import try_answer_schools     # if you added it earlier; else you can remove this line

def try_extract_answer(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic, context-only answers for high-signal intents.
    Returns a single short string or None to fall back to the LLM.
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

    # --- Concepts (optional if you added them)
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

    try:
        ans = try_answer_schools(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    return None
