# extractors/__init__.py
import re
from typing import List, Dict, Any

from .kihon_happo import try_answer_kihon_happo
from .sanshin import try_answer_sanshin
from .schools import try_answer_schools
from .rank import try_answer_rank_striking   # rank-aware kicks/punches/striking

# Plural-friendly token pattern for striking intent in rank questions
_STRIKE_INTENT_RE = re.compile(
    r"\b("
    r"kick|kicks|geri|geris|"
    r"punch|punches|tsuki|tsukis|"
    r"strikes?|striking|"
    r"(?:^|[\s\-])ken\b"          # ...-ken / ' ken ' / line-start ken
    r")\b",
    re.I,
)

_RANK_RE = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)\s+kyu\b", re.I)

def try_extract_answer(question: str, passages: List[Dict[str, Any]]) -> str | None:
    """Route core questions to deterministic extractors; return None if no match."""
    ql = question.lower()

    # ✅ Rank-specific striking (kicks/punches/striking for Nth kyu)
    if _RANK_RE.search(ql) and _STRIKE_INTENT_RE.search(ql):
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
