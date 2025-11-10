# extractors/__init__.py
from typing import List, Dict, Any, Optional

# Leadership (Sokeship) — run first
from .leadership import try_extract_answer as try_leadership

# Rank-specific extractors
from .rank import (
    try_answer_rank_striking,
    try_answer_rank_nage,
    try_answer_rank_jime,
)
# Optional weapons-by-rank (if implemented in rank.py)
try:
    from .rank import try_answer_rank_weapons  # type: ignore
except Exception:
    try_answer_rank_weapons = None  # type: ignore

# Weapons extractor (deterministic, structured)
from .weapons import try_answer_weapons

# Concept extractors
from .kyusho import try_answer_kyusho
from .kihon_happo import try_answer_kihon_happo
from .sanshin import try_answer_sanshin
from .schools import try_answer_schools


def try_extract_answer(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic, context-only answers for high-signal intents.
    Return a short string or None to let the LLM/explainers handle it.
    Order matters: most specific first.
    """
    # Leadership / Soke
    try:
        ans = try_leadership(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # Rank-specific (precise)
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

    # Weapons (deterministic, structured file + rank + glossary)
    try:
        ans = try_answer_weapons(question, passages)
        if ans:
            return ans
    except Exception:
        pass

    # Concepts
    for fn in (try_answer_kyusho, try_answer_kihon_happo, try_answer_sanshin, try_answer_schools):
        try:
            ans = fn(question, passages)
            if ans:
                return ans
        except Exception:
            pass

    return None
