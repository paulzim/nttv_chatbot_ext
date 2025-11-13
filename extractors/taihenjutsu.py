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


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _looks_like_taihen_question(question: str) -> bool:
    """
    Only treat as Taihenjutsu when clearly about ukemi / rolls / taihenjutsu.
    """
    q = _fold(question)
    if "taihenjutsu" in q:
        return True
    if "ukemi" in q or "breakfall" in q or "break fall" in q:
        return True
    if "kaiten" in q or "roll" in q or "rolling" in q:
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


# ----------------- parse Taihenjutsu block -----------------


def _extract_taihen_block() -> str:
    """
    Extract the 'Taihenjutsu- Body Skills' block up to the start of
    'Dakentaijutsu- Striking and Blocking Skills'.
    """
    text = _load_training_text()
    if not text:
        return ""

    start = text.find("Taihenjutsu- Body Skills")
    if start == -1:
        return ""

    end_marker = "Dakentaijutsu- Striking and Blocking Skills"
    end = text.find(end_marker, start)
    if end == -1:
        end = None

    return text[start:end]


def _parse_taihen_records() -> Dict[str, Dict[str, str]]:
    """
    Parse the Taihenjutsu block into a map:
        name -> { "name": ..., "desc": ..., "category": "Ukemi"|"Kaiten" }

    Expecting patterns like:

        Ukemi- Breakfalls
        · Zenpo Ukemi- Forward Breakfall
        ...
        Kaiten- Rolls
        · Zenpo Kaiten Naname- Forward Diagonal Roll
        ...
    """
    block = _extract_taihen_block()
    if not block:
        return {}

    records: Dict[str, Dict[str, str]] = {}
    current_cat: Optional[str] = None

    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Category headings
        if line.startswith("Ukemi-"):
            current_cat = "Ukemi"
            continue
        if line.startswith("Kaiten-"):
            current_cat = "Kaiten"
            continue

        # Bullet lines
        if line.startswith(("·", "-", "*", "•")):
            # Strip bullet
            line_body = line[1:].strip()
            if "-" not in line_body:
                continue
            name_part, desc_part = line_body.split("-", 1)
            name = name_part.strip()
            desc = desc_part.strip()
            if not name:
                continue

            key = _fold(name)
            records[key] = {
                "name": name,
                "desc": desc,
                "category": current_cat or "",
            }

    return records


# ----------------- answering helpers -----------------


def _match_specific_taihen(question: str, records: Dict[str, Dict[str, str]]) -> Optional[str]:
    """
    Try to match a specific ukemi / roll name in the question.
    """
    q = _fold(question)
    for key, rec in records.items():
        if not key:
            continue
        # Whole-word-ish match on the folded name
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, q):
            name = rec["name"]
            desc = rec["desc"]
            cat = rec.get("category") or "Taihenjutsu"
            return f"{name} ({cat}): {desc}"
    return None


def _answer_list_taihen(question: str, records: Dict[str, Dict[str, str]]) -> Optional[str]:
    """
    Handle list-style queries, e.g.:

      - 'list the ukemi breakfalls'
      - 'what are the ukemi?'
      - 'what rolls are in taihenjutsu?'
      - 'what taihenjutsu rolls do we learn?'
    """
    if not records:
        return None

    q = _fold(question)

    # Broader list-intent detection:
    wants_list = (
        "list" in q
        or "what are" in q
        or "which" in q
        or "name the" in q
        # e.g. "what rolls are in taihenjutsu?"
        or ("what" in q and "roll" in q)
        # e.g. "what ukemi do we learn?"
        or ("what" in q and "ukemi" in q)
        # e.g. "what taihenjutsu skills"
        or ("what" in q and "taihenjutsu" in q)
    )
    if not wants_list:
        return None

    # Decide which subset they want: ukemi, rolls, or all
    want_ukemi = "ukemi" in q or "breakfall" in q or "break fall" in q
    want_rolls = "kaiten" in q or "roll" in q or "rolling" in q

    names_ukemi = [rec["name"] for rec in records.values() if rec.get("category") == "Ukemi"]
    names_rolls = [rec["name"] for rec in records.values() if rec.get("category") == "Kaiten"]

    if want_ukemi and names_ukemi:
        return f"Ukemi (breakfalls): {join_oxford(names_ukemi)}"

    if want_rolls and names_rolls:
        return f"Kaiten (rolls): {join_oxford(names_rolls)}"

    # Fallback: list everything if they just said "list the taihenjutsu"
    all_names = [rec["name"] for rec in records.values()]
    if all_names:
        return f"Taihenjutsu skills: {join_oxford(all_names)}"

    return None


# ----------------- public entrypoint -----------------


def try_answer_taihenjutsu(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic Taihenjutsu extractor:

      * Specific skill definitions:
          - 'what is Yoko Nagare?'
          - 'explain Zenpo Kaiten Naname'

      * Lists:
          - 'list the ukemi breakfalls'
          - 'what rolls are in taihenjutsu?'
          - 'what ukemi do we learn in taihenjutsu?'
    """
    if not _looks_like_taihen_question(question):
        return None

    records = _parse_taihen_records()
    if not records:
        return None

    # 1) List-style questions (ukemi/rolls)
    ans = _answer_list_taihen(question, records)
    if ans:
        return ans

    # 2) Specific named skill
    ans = _match_specific_taihen(question, records)
    if ans:
        return ans

    return None
