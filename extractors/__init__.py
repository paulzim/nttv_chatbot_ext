# extractors/__init__.py
from typing import List, Dict, Any, Optional

# ----- Rank-specific extractors (most precise; run first) -----
from .rank import (
    try_answer_rank_striking,
    try_answer_rank_nage,
    try_answer_rank_jime,
    # keep if you've implemented weapons in rank.py; otherwise remove this import + call
    try_answer_rank_weapons,  # optional
)

# ----- Concept extractors (deterministic, context-only) -----
from .kyusho import try_answer_kyusho
from .kihon_happo import try_answer_kihon_happo
from .sanshin import try_answer_sanshin
from .schools import try_answer_schools

# NEW: leadership (current sōke/grandmaster)
from .leadership import try_extract_answer as try_leadership


def try_extract_answer(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic, context-only answers for high-signal intents.
    Return a single short string or None to fall back to the LLM.
    Order matters: most specific (rank) -> leadership -> core concepts.
    """

    ql = question.lower()

    # ---------- Rank-specific ----------
    ans = try_answer_rank_striking(question, passages)
    if ans:
        return ans

    ans = try_answer_rank_nage(question, passages)
    if ans:
        return ans

    ans = try_answer_rank_jime(question, passages)
    if ans:
        return ans

    # Optional: rank weapons
    try:
        ans = try_answer_rank_weapons(question, passages)  # remove if not using
        if ans:
            return ans
    except Exception:
        pass

    # ---------- Leadership (Sōke / Grandmaster) ----------
    # Run BEFORE generic school/sanshin/kyusho so we short-circuit with a person’s name.
    try:
        ans = try_leadership(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # ---------- Concepts ----------
    # Kyusho first so it doesn’t get overshadowed by other concept extractors
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
        ans = try_answer_sansin(question, passages)  # NOTE: keep exact name if your file uses 'sanshin'
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
