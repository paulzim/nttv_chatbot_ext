# extractors/__init__.py
import re
from typing import List, Dict, Any

from .kihon_happo import try_answer_kihon_happo
from .sanshin import try_answer_sanshin
from .schools import try_answer_schools
from .rank import try_answer_rank_striking  # kicks/punches/striking by rank

# Very tolerant detectors
_RANK_ANY   = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)\s+kyu\b", re.I)
_STRIKE_ANY = re.compile(
    r"\b("
    r"kicks?|geri|geris|"
    r"punch(?:es)?|tsuki|tsukis|uraken|"
    r"strikes?|striking|"
    r"(?:^|[\s\-])ken\b"   # matches “-ken” terms as a word
    r")\b",
    re.I,
)

def try_extract_answer(question: str, passages: List[Dict[str, Any]]) -> str | None:
    """Route core questions to deterministic extractors; return None if no match."""
    ql = question.lower()

    # ✅ Always try rank striking first if the question mentions a rank and any striking term
    if _RANK_ANY.search(ql) and _STRIKE_ANY.search(ql):
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

    # 👉 Safety net: if “kyu” appears at all, still try the rank extractor (helps looser queries)
    if "kyu" in ql:
        ans = try_answer_rank_striking(question, passages)
        if ans:
            return ans

    return None

__all__ = ["try_extract_answer"]
