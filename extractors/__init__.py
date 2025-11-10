# extractors/__init__.py
from typing import List, Dict, Any, Optional

# --- Leadership (Sōke/Grandmaster) ---
# Keep leadership early so direct sōke questions short-circuit cleanly.
from .leadership import try_extract_answer as try_leadership

# --- Rank-specific extractors (most precise; run first) ---
from .rank import (
    try_answer_rank_striking,
    try_answer_rank_nage,
    try_answer_rank_jime,
)
# Optional: only if you actually implemented this in rank.py
try:
    from .rank import try_answer_rank_weapons  # type: ignore
except Exception:
    try_answer_rank_weapons = None  # type: ignore

# --- Weapons (deterministic, from NTTV Weapons Reference + rank refs + glossary) ---
# These two cover “at what rank do I learn X?” and “tell me about X”
try:
    from .weapons import (
        try_answer_weapon_rank,
        try_answer_weapon_profile,
    )
except Exception:
    def try_answer_weapon_rank(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
        return None
    def try_answer_weapon_profile(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
        return None

# --- Concepts (deterministic, context-only) ---
from .kyusho import try_answer_kyusho
from .kihon_happo import try_answer_kihon_happo
from .sanshin import try_answer_sanshin
from .schools import try_answer_schools


def try_extract_answer(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic, context-only answers for high-signal intents.
    Return a short string (final answer) or None to let the LLM/explainers handle it.

    Priority:
      1) Rank-specific (striking/nage/jime/[optional rank-weapons])
      2) Leadership (sōke/grandmaster)
      3) Weapons (rank lookup, then profile)
      4) Core concepts (kyusho, kihon happo, sanshin, schools)
    """

    # ----- 1) Rank-specific -----
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

    # ----- 2) Leadership (Sōke/Grandmaster) -----
    try:
        ans = try_leadership(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # ----- 3) Weapons (deterministic) -----
    # 3a) “At what rank do I learn X?”
    try:
        ans = try_answer_weapon_rank(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # 3b) “Tell me about <weapon>”
    try:
        ans = try_answer_weapon_profile(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # ----- 4) Core concepts -----
    for fn in (try_answer_kyusho, try_answer_kihon_happo, try_answer_sanshin, try_answer_schools):
        try:
            ans = fn(question, passages)
            if ans:
                return ans
        except Exception:
            pass

    return None
