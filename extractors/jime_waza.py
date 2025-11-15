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


def _data_dir() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent / "data"


def _load_training_text() -> str:
    """
    Load the NTTV training reference text, which contains the Jime Waza block.
    """
    p = _data_dir() / "nttv training reference.txt"
    try:
        if p.exists():
            return p.read_text(encoding="utf-8")
    except Exception:
        return ""
    return ""


# ----------------- parse Jime Waza block -----------------


def _parse_jime_waza() -> Dict[str, Dict[str, str]]:
    """
    Parse the 'Jime Waza− “Choking” Waza' section from the NTTV
    training reference into a dictionary:

        key (folded name) -> {
            "name": original name,
            "translation": short English gloss,
        }
    """
    text = _load_training_text()
    if not text:
        return {}

    records: Dict[str, Dict[str, str]] = {}

    in_jime = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Heading line – we look for 'Jime Waza' anywhere in the line.
        if "jime waza" in _fold(line):
            in_jime = True
            continue

        # Once we're in the Jime block, we stop at the next obvious section
        # heading. We keep this simple: a non-bullet line that does NOT
        # contain 'jime' is treated as a new section.
        if in_jime and not line.startswith("·") and not line.startswith("-"):
            # We've likely reached the next section (e.g. another topic)
            break

        if in_jime:
            # Expect bullet lines like:
            #   · Hon Jime−Base Choke
            #   · Gyaku Jime− Reverse Choke
            if line.startswith("·") or line.startswith("-"):
                bullet = line.lstrip("·-").strip()
            else:
                bullet = line

            # Normalize the dash variants
            parts = re.split(r"[–\-−]+", bullet, maxsplit=1)
            if not parts:
                continue

            name = parts[0].strip()
            if not name:
                continue

            translation = ""
            if len(parts) > 1:
                translation = parts[1].strip().strip("“”\"'")

            key = _fold(name)
            if key not in records:
                records[key] = {"name": name, "translation": translation}

    return records


# ----------------- intent helpers -----------------


def _looks_like_jime_list_query(q: str) -> bool:
    fq = _fold(q)
    if "jime waza" in fq:
        return True
    if "choking waza" in fq:
        return True
    if "chokes" in fq and ("curriculum" in fq or "list" in fq or "what are" in fq):
        return True
    if "what jime" in fq:
        return True
    return False


def _extract_specific_jime_name(q: str, records: Dict[str, Dict[str, str]]) -> Optional[Dict[str, str]]:
    """
    Try to find a specific Jime name mentioned in the question.
    """
    fq = _fold(q)
    for key, rec in records.items():
        name_folded = _fold(rec["name"])
        if name_folded in fq:
            return rec
    return None


# ----------------- answer formatters -----------------


def _answer_jime_list(records: Dict[str, Dict[str, str]]) -> Optional[str]:
    if not records:
        return None
    names = [rec["name"] for rec in records.values()]
    if not names:
        return None
    return "Jime Waza chokes in the curriculum: " + join_oxford(sorted(names))


def _answer_specific_jime(rec: Dict[str, str]) -> str:
    name = rec["name"]
    translation = rec.get("translation", "")
    if translation:
        return f"{name}: {translation} (Jime Waza choking technique)."
    return f"{name} is one of the Jime Waza choking techniques."


# ----------------- public entrypoint -----------------


def try_answer_jime_waza(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic extractor for Jime Waza (choking techniques) from the
    NTTV training reference.

    Handles:
      - 'what chokes are in the curriculum?'
      - 'what jime waza do we study?'
      - 'in jime waza, what is Sankaku Jime?'
      - 'explain Gyaku Jime'
    """
    records = _parse_jime_waza()
    if not records:
        return None

    # 1) Generic list-style questions
    if _looks_like_jime_list_query(question):
        ans = _answer_jime_list(records)
        if ans:
            return ans

    # 2) Specific named Jime technique
    rec = _extract_specific_jime_name(question, records)
    if rec:
        return _answer_specific_jime(rec)

    return None
