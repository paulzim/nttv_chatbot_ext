# extractors/kyusho.py
# Deterministic Kyusho extractor
# - Only triggers on explicit kyusho / pressure-point questions
# - Parses entries from retrieved passages + full KYUSHO.txt on disk
# - Returns either:
#     * a one-line location/description for a specific point, or
#     * a concise list of points when explicitly asked to list them.

from __future__ import annotations
import re
import unicodedata
from pathlib import Path
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


def _load_full_kyusho_file() -> str:
    """
    Read the full KYUSHO.txt from disk so we don't depend on the retriever
    grabbing the exact chunk that contains a given point.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "data" / "KYUSHO.txt",  # repo layout: root/data/KYUSHO.txt
        here.parent / "KYUSHO.txt",                  # fallback: same dir as this file
    ]
    for p in candidates:
        try:
            if p.exists():
                return p.read_text(encoding="utf-8")
        except Exception:
            continue
    return ""


def _gather_kyusho_text(passages: List[Dict[str, Any]]) -> str:
    """
    Concatenate KYUSHO-related passages *and* append the full file contents
    if available.
    """
    buf: List[str] = []
    for p in passages:
        src = _fold(p.get("source") or "")
        if "kyusho" in src:
            buf.append(p.get("text", ""))

    full_file = _load_full_kyusho_file()
    if full_file.strip():
        buf.append(full_file)

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
        if line.startswith(("-", "*", "•")):
            line = line[1:].strip()

        # Must have a colon to be considered "Name: desc"
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
    """
    Return the folded key of the first kyusho name mentioned in the question.

    Uses word-boundary regex so we don't mistakenly match 'in' from 'points'.
    """
    q = _fold(question)
    for key in points.keys():
        if not key:
            continue
        # Require whole-word match for the key
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, q):
            return key
    return None


def try_answer_kyusho(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic kyusho extractor.

    - Only triggers when the question clearly references kyusho / pressure points.
    - Uses KYUSHO.txt (retrieved chunks + full file) to answer:
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

    # --- 1) List-style queries take precedence over single-point detection ---
    is_list = "list" in q or ("what" in q and "points" in q)
    if is_list:
        names = dedupe_preserve(list(points.keys()))
        if not names:
            return None
        # Use folded names capitalized for display; cap to ~20 for readability
        display_names = [
            " ".join(w.capitalize() for w in n.split())
            for n in names[:20]
        ]
        return f"Kyusho points: {join_oxford(display_names)}"

    # --- 2) Specific point queries ---
    key = _match_point_name(question, points)
    if key:
        desc = points.get(key, "").strip()
        name_display = " ".join(w.capitalize() for w in key.split())
        if desc:
            return f"{name_display}: {desc}"
        else:
            return (
                f"{name_display}: (location/description not listed in the provided context)."
            )

    # Otherwise, let upstream handlers / LLM try
    return None


# Backwards-compat alias for the router, if it imports try_kyusho
def try_kyusho(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    return try_answer_kyusho(question, passages)
