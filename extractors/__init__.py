# extractors/__init__.py
from typing import List, Dict, Any, Optional

# --- Leadership (Sōke/Grandmaster) — keep this near the top in the dispatcher order
from .leadership import try_extract_answer as try_leadership

# --- Rank-specific extractors (precise)
from .rank import (
    try_answer_rank_striking,
    try_answer_rank_nage,
    try_answer_rank_jime,
)
# Optional: only if you've actually implemented this in rank.py; otherwise set to None
try:
    from .rank import try_answer_rank_weapons  # type: ignore
except Exception:
    try_answer_rank_weapons = None  # type: ignore

# --- Weapons (deterministic, from NTTV Weapons Reference + glossary)
try:
    from .weapons import try_answer_weapons
except Exception:
    def try_answer_weapons(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
        return None

# --- Concepts
from .kyusho import try_answer_kyusho
from .kihon_happo import try_answer_kihon_happo
from .sanshin import try_answer_sanshin
from .schools import try_answer_schools


def try_extract_answer(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic, context-only answers for high-signal intents.
    Return a short string or None to let the LLM/explainers handle it.
    Order matters: rank -> leadership -> weapons -> core concepts.
    """

    # ----- Rank-specific -----
    for fn in (try_answer_rank_striking, try_answer_rank_nage, try_answer_rank_jime):
        try:
            ans = fn(question, passages)
            if ans:
                return ans
        except Exception:
            pass

    if try_answer_rank_weapons:
        try:
            ans = try_answer_rank_weapons(question, passages)  # type: ignore
            if ans:
                return ans
        except Exception:
            pass

    # ----- Leadership (Sōke) -----
    try:
        ans = try_leadership(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # ----- Weapons (deterministic) -----
    try:
        ans = try_answer_weapons(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # ----- Concepts -----
    for fn in (try_answer_kyusho, try_answer_kihon_happo, try_answer_sanshin, try_answer_schools):
        try:
            ans = fn(question, passages)
            if ans:
                return ans
        except Exception:
            pass

    return None
