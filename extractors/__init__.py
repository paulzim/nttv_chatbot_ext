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
except Exception:
    def try_answer_rank_weapons(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
        return None

# ----- Deterministic concept/technique extractors
from .kyusho import try_answer_kyusho
from .kihon_happo import try_answer_kihon_happo     # <— MUST be imported
from .techniques import try_answer_technique
from .sanshin import try_answer_sanshin

# Leadership (Soke lookups)
from .leadership import try_extract_answer as try_leadership


def try_extract_answer(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic, context-only answers for high-signal intents.
    Return a short string or None to fall back to the LLM/generic path.
    Order matters: most specific first.
    """

    # --- Rank-specific: Striking / Throws / Chokes
    ans = try_answer_rank_striking(question, passages)
    if ans: return ans

    ans = try_answer_rank_nage(question, passages)
    if ans: return ans

    ans = try_answer_rank_jime(question, passages)
    if ans: return ans

    # --- Rank-specific: Requirements (ENTIRE block for “requirements for X kyu”)
    ans = try_answer_rank_requirements(question, passages)
    if ans: return ans

    # --- Rank-specific: Weapons by rank (optional)
    ans = try_answer_rank_weapons(question, passages)
    if ans: return ans

    # --- Concept: Kyusho (short, deterministic)
    ans = try_answer_kyusho(question, passages)
    if ans: return ans

    # --- Kihon Happo (run BEFORE techniques so it wins over general technique matches)
    ans = try_answer_kihon_happo(question, passages)
    if ans: return ans

    # --- Techniques (Omote Gyaku, Musha Dori, Jumonji no Kata, etc.)
    ans = try_answer_technique(question, passages)
    if ans: return ans

    # --- Concept: Sanshin
    ans = try_answer_sanshin(question, passages)
    if ans: return ans

    # --- Leadership (Soke / headmaster)
    ans = try_leadership(question, passages)
    if ans: return ans

    return None
