# extractors/kyusho.py
# Deterministic Kyusho extractor
# - Only triggers on explicit kyusho / pressure-point questions
# - Parses entries from retrieved passages only (no filesystem access)
# - Returns either a one-line location/description for a specific point,
#   or a concise list of points when explicitly asked to list them.

from __future__ import annotations
import re
import unicodedata
from typing import List, Dict, Any, Optional
from .common import dedupe_preserve, join_oxford


def _fold(s: str) -> str:
    """Case- and accent-insensitive fold."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()


def _looks_like_kyusho_question(question: str) -> bool:
    """
    Only treat this as a kyusho question when the user clearly references
    kyusho / pressure points.

    This avoids stealing technique questions like 'describe Oni Kudaki'.
    """
    q = _fold(question)
    return (
        "kyusho" in q
        or "pressure point" in q
        or "pressure points" in q
    )


def _gather_kyusho_text(passages: List[Dict[str, Any]]) -> str:
    """Concatenate KYUSHO-related passages."""
    buf: List[str] = []
    for p in passages:
        src = _fold(p.get("source") or "")
        if "kyusho" in src:
            buf.append(p.get("text", ""))
    return "\n".join(buf)


def _parse_points(text: str) -> Dict[str, str]:
    """
    Very simple parser for KYUSHO.txt.

    We look for lines of the form:

        NAME: description...

    or bullet variants:

        - Name: description...

    and build a {folded_name -> description} mapping.
    """
    points: Dict[str, str] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue

        # Strip bullets if present
        if line.startswith(("-", "*")):
            line = line[1:].strip()

        if ":" not in line:
            continue

        name, desc = line.split(":", 1)
        name = name.strip()
        desc = desc.strip()
        if not name:
            continue

        key = _fold(name)
        # Prefer the first occurrence; later duplicates can be ignored
        if key not in points:
            points[key] = desc

    return points


def _match_point_name(question: str, points: Dict[str, str]) -> Optional[str]:
    """Return the folded key of the first kyusho name mentioned in the question."""
    q = _fold(question)
    for key in points.keys():
        # kyusho names are short (1–2 tokens): 'ura kimon', 'kasumi', 'suigetsu', etc.
        if key and key in q:
            return key
    return None


def try_answer_kyusho(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic kyusho extractor.

    - Only triggers when the question clearly references kyusho / pressure points.
    - Uses KYUSHO.txt to answer:
        * 'Where is Ura Kimon kyusho?' style questions
        * 'List the kyusho pressure points' style questions
    """
    if not _looks_like_kyusho_question(question):
        return None

    text = _gather_kyusho_text(passages)
    if not text.strip():
        return None

    points = _parse_points(text)
    if not points:
        return None

    q = _fold(question)
    key = _match_point_name(question, points)

    # If the user clearly asked about a specific point and we can find it
    if key:
        desc = points.get(key, "").strip()
        name_display = " ".join(w.capitalize() for w in key.split())
        if desc:
            return f"{name_display}: {desc}"
        else:
            return (
                f"{name_display}: (location/description not listed in the provided context)."
            )

    # No specific point matched: check for list-style queries
    if "list" in q or ("what" in q and "points" in q):
        names = dedupe_preserve([k for k in points.keys()])
        if not names:
            return None
        # Use the raw names (unfolded) for display where possible
        display_names = [
            " ".join(w.capitalize() for w in n.split()) for n in names[:20]
        ]
        return join_oxford(display_names)

    # Otherwise, let upstream handlers try
    return None


# Backwards-compat alias for the router, if it imports try_kyusho
def try_kyusho(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    return try_answer_kyusho(question, passages)
