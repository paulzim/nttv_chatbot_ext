# extractors/technique_diff.py
"""
Deterministic "difference between techniques" extractor.

Answers questions like:
- "What is the difference between Omote Gyaku and Ura Gyaku?"
- "Omote Gyaku vs Ura Gyaku"
- "Compare Musha Dori and Oni Kudaki"

Implementation:
- Uses Technique Descriptions.md via technique_loader.parse_technique_md.
- Builds indexes with technique_loader.build_indexes.
- Resolves each technique name and then formats a structured comparison.
"""

from __future__ import annotations

import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Any, Optional

from .technique_loader import parse_technique_md, build_indexes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _same_source_name(p_source: str, target_name: str) -> bool:
    if not p_source:
        return False
    return os.path.basename(p_source).lower() == os.path.basename(target_name).lower()


def _gather_technique_md(passages: List[Dict[str, Any]]) -> str:
    """
    Collect Technique Descriptions text from retrieved passages,
    falling back to data/Technique Descriptions.md if needed.
    """
    chunks: List[str] = []

    for p in passages:
        src_raw = p.get("source") or ""
        if _same_source_name(src_raw, "Technique Descriptions.md") or "technique descriptions" in src_raw.lower():
            txt = p.get("text", "")
            if txt:
                chunks.append(txt)

    if chunks:
        return "\n".join(chunks)

    # Fallback: read from data/Technique Descriptions.md on disk
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "data" / "Technique Descriptions.md",
        here.parent / "Technique Descriptions.md",
    ]
    for p in candidates:
        try:
            if p.exists():
                return p.read_text(encoding="utf-8")
        except Exception:
            continue

    return ""


def _build_indexes_from_md(md_text: str) -> Optional[Dict[str, Any]]:
    if not md_text.strip():
        return None
    records = parse_technique_md(md_text)
    if not records:
        return None
    return build_indexes(records)


def _resolve_technique_name(
    cand_raw: str, indexes: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Resolve a candidate technique name to a record using the indexes.
    Uses:
      - by_lower (exact lowercase match)
      - fuzzy match over canonical names as fallback
    """
    if not cand_raw:
        return None

    cand = cand_raw.strip()
    if not cand:
        return None

    by_name = indexes.get("by_name", {})
    by_lower = indexes.get("by_lower", {})

    low = cand.lower()
    if low in by_lower:
        canon_name = by_lower[low]
        rec = by_name.get(canon_name)
        if rec:
            return rec

    # Fuzzy comparison against canonical names
    best_rec = None
    best_score = 0.0
    for name, rec in by_name.items():
        score = SequenceMatcher(None, low, name.lower()).ratio()
        if score > best_score:
            best_score = score
            best_rec = rec

    if best_rec and best_score >= 0.75:
        return best_rec

    return None


def _looks_like_diff_question(question: str) -> bool:
    q = question.lower()
    return any(
        tok in q
        for tok in ["difference between", "different from", "diff between", " vs ", "versus", "compare "]
    )


def _extract_pair(question: str) -> Optional[tuple[str, str]]:
    """
    Try to extract "A" and "B" from questions like:
    - "What's the difference between Omote Gyaku and Ura Gyaku?"
    - "Compare Musha Dori and Oni Kudaki"
    - "Omote Gyaku vs Ura Gyaku"
    """
    q = question.strip().rstrip("?.! ")

    # 1) "difference between A and B"
    m = re.search(r"difference between\s+(.+?)\s+and\s+(.+)", q, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # 2) "compare A and B"
    m = re.search(r"compare\s+(.+?)\s+and\s+(.+)", q, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # 3) "A vs B" / "A versus B"
    m = re.search(r"(.+?)\s+vs\.?\s+(.+)", q, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    m = re.search(r"(.+?)\s+versus\s+(.+)", q, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    return None


def _fmt_bool(v: Any) -> str:
    if v is True:
        return "Yes"
    if v is False:
        return "No"
    return "—"


def _format_diff(rec1: Dict[str, Any], rec2: Dict[str, Any]) -> str:
    """
    Format a structured comparison between two techniques.
    All content is directly derived from Technique Descriptions.md.
    """
    name1 = rec1.get("name") or "Technique 1"
    name2 = rec2.get("name") or "Technique 2"

    fields = [
        ("Translation", "translation"),
        ("Type", "type"),
        ("Rank intro", "rank"),
        ("Primary focus", "primary_focus"),
        ("Safety", "safety"),
        ("Partner required", "partner_required"),
        ("Solo", "solo"),
        ("Description", "description"),
    ]

    lines: List[str] = []
    lines.append(f"Difference between {name1} and {name2}:")

    for label, key in fields:
        v1 = rec1.get(key)
        v2 = rec2.get(key)

        if key in ("partner_required", "solo"):
            v1 = _fmt_bool(v1)
            v2 = _fmt_bool(v2)
        else:
            v1 = (v1 or "").strip()
            v2 = (v2 or "").strip()

        if not v1 and not v2:
            continue

        lines.append(f"\n{label}:")
        lines.append(f"- {name1}: {v1 if v1 else '—'}")
        lines.append(f"- {name2}: {v2 if v2 else '—'}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def try_answer_technique_diff(
    question: str, passages: List[Dict[str, Any]]
) -> Optional[str]:
    """
    Deterministic comparison of two named techniques.
    Returns None if the question isn't clearly a diff / comparison request,
    or if we can't confidently resolve both technique names.
    """
    if not _looks_like_diff_question(question):
        return None

    pair = _extract_pair(question)
    if not pair:
        return None

    left_raw, right_raw = pair

    md_text = _gather_technique_md(passages)
    indexes = _build_indexes_from_md(md_text)
    if not indexes:
        return None

    rec1 = _resolve_technique_name(left_raw, indexes)
    rec2 = _resolve_technique_name(right_raw, indexes)
    if not rec1 or not rec2:
        return None

    return _format_diff(rec1, rec2)
