# extractors/__init__.py
import re
from typing import List, Dict, Any

from .kihon_happo import try_answer_kihon_happo
from .sanshin import try_answer_sanshin
from .schools import try_answer_schools
from .rank import try_answer_rank_striking   # NEW

def try_extract_answer(question: str, passages: List[Dict[str, Any]]) -> str | None:
    """Route core questions to deterministic extractors; return None if no match."""
    ql = question.lower()

    # Rank-specific striking (kicks/punches/striking for Nth kyu)
    if re.search(r"\b\d{1,2}(?:st|nd|rd|th)\s+kyu\b", ql) and re.search(r"\b(kick|geri|punch|tsuki|ken|strike|striking)\b", ql):
        ans = try_answer_rank_striking(question, passages)
        if ans:
            return ans

    # Kihon Happo
    if re.search(r"\bkihon\s+happo\b", ql):
        ans = try_answer_kihon_happo(passages)
        if ans:
            return ans

    # Sanshin (various spellings)
    if re.search(r"\bsanshin(?:\s+no\s+kata)?\b", ql) or re.search(r"\bsan\s+shin\b", ql):
        ans = try_answer_sanshin(passages)
        if ans:
            return ans

    # Schools of the Bujinkan
    if "bujinkan" in ql and re.search(r"\bschool", ql):
        ans = try_answer_schools(passages)
        if ans:
            return ans

    return None

__all__ = ["try_extract_answer"]
