# extractors/__init__.py
from typing import List, Dict, Any, Optional

# ----- Rank-specific extractors (most precise; run first)
from .rank import (
    try_answer_rank_striking,
    try_answer_rank_nage,
    try_answer_rank_jime,
    try_answer_rank_requirements,   # explicit “requirements for X kyu”
)

# If you implemented rank weapons in rank.py, import it; otherwise provide a no-op.
try:
    from .rank import try_answer_rank_weapons  # optional
except Exception:  # pragma: no cover
    def try_answer_rank_weapons(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
        return None

# Kyusho (pressure points)
from .kyusho import try_answer_kyusho

# Kihon Happo
from .kihon_happo import try_answer_kihon_happo

# Kamae (stances)
from .kamae import try_answer_kamae

# Single-technique CSV extractor
from .techniques import try_answer_technique

# Sanshin no Kata
from .sanshin import try_answer_sanshin

# Leadership / sōke / shihan info
from .leadership import try_extract_answer as try_leadership

# Taihenjutsu
from .taihenjutsu import try_answer_taihenjutsu

# Dakentaijutsu
from .dakentaijutsu import try_answer_dakentaijutsu



def try_extract_answer(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic dispatcher.

    The order here is important: we prefer the most specific / rank-tied
    answers first, then kyusho, then core curriculum concepts (kihon, kamae),
    then techniques, and finally leadership.
    """

    # --- Rank: weapons at a given rank (if implemented inside rank.py)
    ans = try_answer_rank_weapons(question, passages)
    if ans:
        return ans

    # --- Rank: striking / kicks / punches
    ans = try_answer_rank_striking(question, passages)
    if ans:
        return ans

    # --- Rank: throws / nage waza
    ans = try_answer_rank_nage(question, passages)
    if ans:
        return ans

    # --- Rank: jime (chokes) at a given rank
    ans = try_answer_rank_jime(question, passages)
    if ans:
        return ans

    # --- Rank: full requirements block (“requirements for X kyu”)
    ans = try_answer_rank_requirements(question, passages)
    if ans:
        return ans

    # --- Kyusho (pressure points)
    ans = try_answer_kyusho(question, passages)
    if ans:
        return ans

    # --- Concept: Kihon Happo (core 8)
    ans = try_answer_kihon_happo(question, passages)
    if ans:
        return ans
    
    # --- Taihenjutsu (ukemi / rolls)
    ans = try_answer_taihenjutsu(question, passages)
    if ans:
        return ans

    # --- Kamae (stances: rank, weapon, specific kamae)
    ans = try_answer_kamae(question, passages)
    if ans:
        return ans

    ans = try_answer_dakentaijutsu(question, passages)
    if ans:
        return ans

    # --- Single-technique definitions from Technique Descriptions.md
    ans = try_answer_technique(question, passages)
    if ans:
        return ans

    # --- Concept: Sanshin no Kata
    ans = try_answer_sanshin(question, passages)
    if ans:
        return ans

    # --- Leadership (Sōke / headmaster, shihan roles)
    ans = try_leadership(question, passages)
    if ans:
        return ans

    return None
