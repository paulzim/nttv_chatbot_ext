from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import unicodedata
import re

from .common import join_oxford


# ----------------- small helpers -----------------


def _fold(s: str) -> str:
    """Case- and accent-insensitive fold."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()


def _looks_like_nage_question(question: str) -> bool:
    """
    Fire only when it’s clearly about Nage Waza / throwing waza, so we
    don’t collide with general technique or rank questions.

    Triggers:
      - 'nage waza'
      - 'throwing waza'
      - 'throwing techniques' + (nage/bujinkan/curriculum)
      - 'throws' + 'nage'
      - 'in nage waza, what is ...'
    """
    q = _fold(question)

    if "nage waza" in q:
        return True
    if "throwing waza" in q:
        return True
    if "throwing techniques" in q and ("nage" in q or "bujinkan" in q or "curriculum" in q):
        return True
    if "throws" in q and "nage" in q:
        return True

    return False


# ----------------- file loading -----------------


def _data_dir() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent / "data"


def _load_training_text() -> str:
    p = _data_dir() / "nttv training reference.txt"
    try:
        if p.exists():
            return p.read_text(encoding="utf-8")
    except Exception:
        return ""
    return ""


# ----------------- parse Nage Waza sections -----------------


def _parse_nage_records() -> Dict[str, Dict[str, str]]:
    """
    Parse all Nage Waza sections from the NTTV training reference into:

        key (folded name) -> {
            "name": original name,
            "desc": description from the line,
            "group": heading descriptor (Throwing Techniques / Throwing Waza / Nage Waza / (cont.))
        }

    We look for headings like:
        Nage waza- Throwing Techniques
        Nage Waza− Throwing Waza
        Nage Waza (cont.)

    And collect bullet lines under each heading until the section ends.
    """
    text = _load_training_text()
    if not text:
        return {}

    lines = text.splitlines()
    records: Dict[str, Dict[str, str]] = {}

    in_nage = False
    current_group = "Nage Waza"

    for raw in lines:
        line = raw.strip()

        # Start of any Nage Waza section
        if line.startswith("Nage waza") or line.startswith("Nage Waza"):
            in_nage = True

            # Extract descriptor after the first dash-like character, if present
            parts = re.split(r"[-–−]", line, 1)
            group_desc = parts[1].strip() if len(parts) == 2 else ""
            current_group = group_desc or "Nage Waza"
            continue

        if not in_nage:
            continue

        # End of the current Nage block on blank or non-bullet line
        if not line:
            in_nage = False
            continue

        if not line.startswith(("·", "-", "*", "•")):
            # Non-bullet means we've left the Nage Waza section
            in_nage = False
            continue

        # Bullet line inside a Nage Waza section
        body = line[1:].strip()
        if not body:
            continue

        # Split "Name− Description" or "Name- Description" using dash-like chars
        parts = re.split(r"[-–−]", body, 1)
        if len(parts) == 2:
            name_part, desc_part = parts
            name = name_part.strip()
            desc = desc_part.strip()
        else:
            name = body.strip()
            desc = ""

        if not name:
            continue

        key = _fold(name)
        if key not in records:
            records[key] = {
                "name": name,
                "desc": desc,
                "group": current_group,
            }

    return records


# ----------------- answering helpers -----------------


def _answer_nage_list(question: str, records: Dict[str, Dict[str, str]]) -> Optional[str]:
    """
    Handle list-style questions, e.g.:

      - 'what are the nage waza?'
      - 'list the nage waza throws'
      - 'what throws are in nage waza?'
    """
    if not records:
        return None

    q = _fold(question)

    wants_list = (
        "list" in q
        or "what are" in q
        or "which" in q
        or "name the" in q
        or ("what" in q and "throws" in q)
        or ("what" in q and "nage waza" in q)
    )
    if not wants_list:
        return None

    names = [rec["name"] for rec in records.values()]
    if not names:
        return None

    return f"Nage Waza throws: {join_oxford(names)}"


def _answer_specific_nage(question: str, records: Dict[str, Dict[str, str]]) -> Optional[str]:
    """
    Answer questions about a specific throw when the context clearly
    references Nage Waza, e.g.:

      - 'in nage waza, what is Harai Goshi?'
      - 'explain Tomoe Nage in the nage waza list'
    """
    q = _fold(question)

    for key, rec in records.items():
        if not key:
            continue

        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, q):
            name = rec["name"]
            desc = rec["desc"]
            group = rec.get("group") or "Nage Waza"
            if desc:
                return f"{name} ({group}): {desc}"
            else:
                return f"{name} ({group})."

    return None


# ----------------- public entrypoint -----------------


def try_answer_nage_waza(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic Nage Waza extractor:

      * Lists:
          - 'what are the nage waza?'
          - 'list the nage waza throws'
          - 'what throws are in nage waza?'

      * Specific throws (only when Nage Waza context is explicit):
          - 'in nage waza, what is Harai Goshi?'
          - 'explain Tomoe Nage in the nage waza list'

    We intentionally *do not* answer rank-specific questions:
        e.g. 'what throws do I learn at 6th kyu?' -> handled by rank_nage
    """
    if not _looks_like_nage_question(question):
        return None

    # If the question is clearly rank-based, bail out and let rank.py handle it.
    q = _fold(question)
    if re.search(r"\b(\d+)(st|nd|rd|th)\s+kyu\b", q) or re.search(r"\b(\d+)(st|nd|rd|th)\s+dan\b", q) or " rank" in q:
        return None

    records = _parse_nage_records()
    if not records:
        return None

    # 1) List-style questions
    ans = _answer_nage_list(question, records)
    if ans:
        return ans

    # 2) Specific named throw in Nage Waza context
    ans = _answer_specific_nage(question, records)
    if ans:
        return ans

    return None
