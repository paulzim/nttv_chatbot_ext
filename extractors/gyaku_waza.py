# extractors/gyaku_waza.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import unicodedata
import re

from .common import join_oxford

# Prefer the shared technique loader if available
try:
    from .technique_loader import parse_technique_md  # type: ignore
except Exception:  # pragma: no cover - fallback path
    parse_technique_md = None  # type: ignore


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


def _load_technique_md() -> str:
    p = _data_dir() / "Technique Descriptions.md"
    try:
        if p.exists():
            return p.read_text(encoding="utf-8")
    except Exception:
        return ""
    return ""


# ----------------- joint-lock rows loading -----------------

_JOINT_ROWS_CACHE: Optional[List[Dict[str, Any]]] = None
_JOINT_INDEX_CACHE: Optional[Dict[str, Any]] = None


def _fallback_parse_joint_rows(md_text: str) -> List[Dict[str, Any]]:
    """
    Very simple CSV fallback for the Joint Locks section if
    technique_loader.parse_technique_md is not available.

    Assumes lines in the Joint Locks block are plain CSV:
      Name, Japanese, Translation, Type, Rank, In Rank, Primary Focus,
      Safety, Partner Required, Can Train Solo, Tags, Description
    """
    rows: List[Dict[str, Any]] = []
    if not md_text:
        return rows

    in_joints = False
    for raw in md_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# Joint Locks"):
            in_joints = True
            continue
        if in_joints and line.startswith("# "):
            # Reached next major section
            break
        if in_joints:
            if line.startswith("##"):
                # format comment, skip
                continue
            if line.startswith(">"):
                continue
            if "," not in line:
                continue
            parts = [c.strip() for c in line.split(",")]
            if len(parts) < 4:
                continue
            # Pad to 12 fields if needed
            while len(parts) < 12:
                parts.append("")
            name, japanese, translation, typ, rank = parts[:5]
            in_rank = parts[5].strip()
            primary_focus = parts[6].strip()
            safety = parts[7].strip()
            partner_required = parts[8].strip()
            solo = parts[9].strip()
            tags = parts[10].strip()
            desc = ",".join(parts[11:]).strip()
            rows.append(
                {
                    "name": name,
                    "japanese": japanese,
                    "translation": translation,
                    "type": typ,
                    "rank": rank,
                    "in_rank": in_rank,
                    "primary_focus": primary_focus,
                    "safety": safety,
                    "partner_required": partner_required,
                    "solo": solo,
                    "tags": tags,
                    "description": desc,
                }
            )
    return rows


def _load_joint_lock_rows() -> List[Dict[str, Any]]:
    global _JOINT_ROWS_CACHE
    if _JOINT_ROWS_CACHE is not None:
        return _JOINT_ROWS_CACHE

    md_text = _load_technique_md()
    rows: List[Dict[str, Any]] = []

    if parse_technique_md is not None and md_text:
        try:
            rows = parse_technique_md(md_text)
        except Exception:
            rows = _fallback_parse_joint_rows(md_text)
    else:
        rows = _fallback_parse_joint_rows(md_text)

    joint_rows: List[Dict[str, Any]] = []
    for r in rows:
        typ = str(r.get("type", "")).strip().lower()
        if "joint" in typ and "lock" in typ:
            joint_rows.append(r)

    _JOINT_ROWS_CACHE = joint_rows
    return joint_rows


def _build_joint_indexes() -> Dict[str, Any]:
    """
    Build:
      - rows: list of joint-lock rows
      - by_name: canonical name -> row
      - alias_to_name: folded alias -> canonical name
    """
    rows = _load_joint_lock_rows()
    by_name: Dict[str, Dict[str, Any]] = {}
    alias_to_name: Dict[str, str] = {}

    for r in rows:
        name = (r.get("name") or "").strip()
        if not name:
            continue

        # canonical
        by_name[name] = r
        key = _fold(name)
        if key and key not in alias_to_name:
            alias_to_name[key] = name

        # translation & japanese
        for alias in [r.get("translation"), r.get("japanese")]:
            a = (alias or "").strip()
            if a:
                fa = _fold(a)
                if fa and fa not in alias_to_name:
                    alias_to_name[fa] = name

        # tags (pipe separated or list)
        tags_val = r.get("tags") or []
        if isinstance(tags_val, str):
            tags_list = [t.strip() for t in tags_val.split("|") if t.strip()]
        else:
            tags_list = [str(t).strip() for t in tags_val if str(t).strip()]

        for t in tags_list:
            ft = _fold(t)
            if ft and ft not in alias_to_name:
                alias_to_name[ft] = name

    return {
        "rows": rows,
        "by_name": by_name,
        "alias_to_name": alias_to_name,
    }


def _get_joint_indexes() -> Dict[str, Any]:
    global _JOINT_INDEX_CACHE
    if _JOINT_INDEX_CACHE is None:
        _JOINT_INDEX_CACHE = _build_joint_indexes()
    return _JOINT_INDEX_CACHE


# ----------------- question classification -----------------


def _looks_like_gyaku_question(question: str, alias_to_name: Dict[str, str]) -> bool:
    """
    Decide if this looks like a gyaku / joint-lock question.
    """
    q = _fold(question)

    # Obvious category/terminology hints
    if "gyaku waza" in q or "joint lock" in q or "joint locks" in q:
        return True
    if "reversal" in q and ("lock" in q or "joint" in q):
        return True
    if "gyaku" in q:
        return True

    # Contains any known lock name or alias?
    for alias_key in alias_to_name.keys():
        if not alias_key:
            continue
        if re.search(r"\b" + re.escape(alias_key) + r"\b", q):
            return True

    return False


def _extract_rank_from_question(question: str) -> Optional[str]:
    """
    Extract rank phrase from the question, if any.
    e.g. '6th kyu', '3rd kyu', 'shodan'
    """
    q = _fold(question)
    m = re.search(r"\b(\d+)(st|nd|rd|th)?\s*kyu\b", q)
    if m:
        return m.group(0)
    if "shodan" in q:
        return "shodan"
    return None


def _wants_rank_answer(question: str) -> bool:
    q = _fold(question)
    return (
        "what rank" in q
        or "which rank" in q
        or "at what rank" in q
        or ("rank" in q and "learn" in q)
    )


def _wants_list_joint_locks(question: str) -> bool:
    q = _fold(question)
    if not ("gyaku" in q or "joint lock" in q or "joint locks" in q or "locks" in q):
        return False

    return (
        "list" in q
        or "what are" in q
        or "which" in q
        or "name the" in q
        or "show me" in q
    )


# ----------------- formatting helpers -----------------


def _bool_to_yn(val: Any) -> str:
    if isinstance(val, str):
        v = val.strip().lower()
        return "Yes" if v in {"1", "true", "yes", "y", "✅", "✓", "✔"} else "No"
    return "Yes" if val else "No"


def _format_joint_lock(row: Dict[str, Any]) -> str:
    name = (row.get("name") or "").strip()
    japanese = (row.get("japanese") or "").strip()
    translation = (row.get("translation") or "").strip()
    typ = (row.get("type") or "").strip()
    rank = (row.get("rank") or "").strip() or "Not Ranked"
    in_rank = row.get("in_rank")
    primary_focus = (row.get("primary_focus") or "").strip()
    safety = (row.get("safety") or "").strip()
    partner_required = row.get("partner_required")
    solo = row.get("solo")
    tags_val = row.get("tags") or []
    if isinstance(tags_val, str):
        tags = [t.strip() for t in tags_val.split("|") if t.strip()]
    else:
        tags = [str(t).strip() for t in tags_val if str(t).strip()]
    description = (row.get("description") or "").strip()

    lines: List[str] = []

    # Headline
    headline = name
    if translation:
        headline += f" — {translation}"
    if typ:
        headline += f" ({typ})"
    lines.append(headline)

    # Rank
    rank_line = f"Rank Intro: {rank}"
    if in_rank not in (None, ""):
        if isinstance(in_rank, bool):
            if in_rank:
                rank_line += " (in-rank ✅)"
        else:
            # treat non-empty truthy string as in-rank
            v = str(in_rank).strip().lower()
            if v in {"1", "true", "yes", "y", "✅", "✓", "✔"}:
                rank_line += " (in-rank ✅)"
    lines.append(rank_line)

    if japanese:
        lines.append(f"Japanese: {japanese}")
    if primary_focus:
        lines.append(f"Primary Focus: {primary_focus}")
    if safety:
        lines.append(f"Safety: {safety}")
    if partner_required not in (None, ""):
        lines.append(f"Partner Required: {_bool_to_yn(partner_required)}")
    if solo not in (None, ""):
        lines.append(f"Can Train Solo: {_bool_to_yn(solo)}")
    if tags:
        lines.append("Tags: " + " | ".join(tags))

    if description:
        lines.append("")
        lines.append("Definition:")
        lines.append(description)

    return "\n".join(lines)


def _format_rank_for_lock(row: Dict[str, Any]) -> str:
    name = (row.get("name") or "").strip()
    rank = (row.get("rank") or "").strip() or "Not Ranked"
    in_rank = row.get("in_rank")

    extra = ""
    if in_rank not in (None, ""):
        if isinstance(in_rank, bool):
            if in_rank:
                extra = " and is part of the in-rank requirements."
        else:
            v = str(in_rank).strip().lower()
            if v in {"1", "true", "yes", "y", "✅", "✓", "✔"}:
                extra = " and is part of the in-rank requirements."

    return f"{name} is introduced at {rank}{extra}"


# ----------------- answering helpers -----------------


def _find_lock_name_in_question(
    question: str, alias_to_name: Dict[str, str]
) -> Optional[str]:
    q = _fold(question)
    for alias_key, canon_name in alias_to_name.items():
        if not alias_key:
            continue
        if re.search(r"\b" + re.escape(alias_key) + r"\b", q):
            return canon_name
    return None


def _answer_list_locks(rows: List[Dict[str, Any]]) -> Optional[str]:
    if not rows:
        return None

    items: List[str] = []
    for r in rows:
        name = (r.get("name") or "").strip()
        translation = (r.get("translation") or "").strip()
        if name and translation:
            items.append(f"{name} — {translation}")
        elif name:
            items.append(name)

    if not items:
        return None

    return "Gyaku Waza / joint locks in this curriculum: " + "; ".join(items)


def _answer_locks_for_rank(rank_query: str, rows: List[Dict[str, Any]]) -> Optional[str]:
    rq = _fold(rank_query)
    matched: List[str] = []

    for r in rows:
        r_rank = (r.get("rank") or "").strip()
        if not r_rank:
            continue
        if _fold(r_rank) == rq:
            name = (r.get("name") or "").strip()
            translation = (r.get("translation") or "").strip()
            if name and translation:
                matched.append(f"{name} — {translation}")
            elif name:
                matched.append(name)

    if not matched:
        return None

    return f"Joint locks at {rank_query}: " + "; ".join(matched)


# ----------------- public entrypoint -----------------


def try_answer_gyaku_waza(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Unified Gyaku / joint-lock extractor.

    Handles:
      * 'what is Omote Gyaku?'
      * 'explain Musha Dori'
      * 'at what rank do we learn Oni Kudaki?'
      * 'list the gyaku waza'
      * 'what joint locks are in the curriculum?'
      * 'what joint locks are at 6th kyu?'

    All answers are derived from Technique Descriptions.md joint-lock rows.
    """
    idx = _get_joint_indexes()
    rows: List[Dict[str, Any]] = idx["rows"]
    by_name: Dict[str, Dict[str, Any]] = idx["by_name"]
    alias_to_name: Dict[str, str] = idx["alias_to_name"]

    if not rows:
        return None

    if not _looks_like_gyaku_question(question, alias_to_name):
        return None

    # Rank-specific lock list: "what joint locks are at 6th kyu?"
    rank_query = _extract_rank_from_question(question)
    if rank_query and ("joint lock" in _fold(question) or "joint locks" in _fold(question) or "gyaku" in _fold(question)):
        ans = _answer_locks_for_rank(rank_query, rows)
        if ans:
            return ans

    # List all joint locks
    if _wants_list_joint_locks(question):
        ans = _answer_list_locks(rows)
        if ans:
            return ans

    # Specific named lock
    canon = _find_lock_name_in_question(question, alias_to_name)
    if canon:
        row = by_name.get(canon)
        if not row:
            return None

        # Rank-intent for this specific lock
        if _wants_rank_answer(question):
            return _format_rank_for_lock(row)

        # Full profile
        return _format_joint_lock(row)

    return None
