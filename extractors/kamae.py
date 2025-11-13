# extractors/kamae.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import unicodedata
import re
from pathlib import Path

from .common import join_oxford

# ----------------- small helpers -----------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _fold(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()


def _looks_like_kamae_question(question: str) -> bool:
    q = _fold(question)
    # Keep this fairly strict so we don't steal unrelated questions
    return "kamae" in q or "stance" in q or "stances" in q

# ----------------- file loading -----------------

def _data_dir() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent / "data"


def _load_file(name: str) -> str:
    p = _data_dir() / name
    try:
        if p.exists():
            return p.read_text(encoding="utf-8")
    except Exception:
        return ""
    return ""

# ----------------- basic kamae (Technique Descriptions.md) -----------------

EXPECTED_COLS = 12  # same schema as techniques.py


def _split_row_limited(raw: str) -> List[str]:
    parts = raw.split(",", EXPECTED_COLS - 1)
    parts = [p.strip() for p in parts]
    if len(parts) > EXPECTED_COLS:
        head = parts[:EXPECTED_COLS - 1]
        tail = ",".join(parts[EXPECTED_COLS - 1:])
        parts = head + [tail]
    if len(parts) < EXPECTED_COLS:
        parts += [""] * (EXPECTED_COLS - len(parts))
    return parts


def _iter_csv_lines(md_text: str):
    for raw in (md_text or "").splitlines():
        st = raw.strip()
        if not st:
            continue
        if st.startswith("#") or st.startswith("```"):
            continue
        if "," in raw:
            yield raw


def _load_kamae_records() -> Dict[str, Dict[str, Any]]:
    """
    Load all rows where Type == 'Kamae' from Technique Descriptions.md
    and index by folded name (with and without 'no Kamae').
    """
    text = _load_file("Technique Descriptions.md")
    out: Dict[str, Dict[str, Any]] = {}
    if not text:
        return out

    for raw in _iter_csv_lines(text):
        row = _split_row_limited(raw)
        if len(row) < EXPECTED_COLS:
            continue
        rec = {
            "name": row[0],
            "japanese": row[1],
            "translation": row[2],
            "type": row[3],
            "rank": row[4],
            "in_rank": row[5],
            "primary_focus": row[6],
            "safety": row[7],
            "partner_required": row[8],
            "solo": row[9],
            "tags": row[10],
            "description": row[11],
        }
        if _fold(rec["type"]) != "kamae":
            continue

        name = rec["name"] or ""
        key_full = _fold(name)
        out[key_full] = rec

        # Also allow shortened form without "no Kamae"
        if "no Kamae" in name:
            short = name.replace("no Kamae", "").strip()
            if short:
                out[_fold(short)] = rec

    return out


def _format_kamae(rec: Dict[str, Any]) -> str:
    name = rec.get("name") or "Kamae"
    translation = rec.get("translation") or ""
    rank = rec.get("rank") or ""
    desc = (rec.get("description") or "").strip()

    lines = [f"{name}:"]
    if translation:
        lines.append(f"- Translation: {translation}")
    if rank:
        lines.append(f"- Rank intro: {rank}")
    if desc:
        lines.append(f"- Description: {desc}")
    else:
        lines.append("- Description: (not listed).")
    return "\n".join(lines)


def _answer_specific_kamae(question: str) -> Optional[str]:
    """
    Handle questions like 'what is Hicho no Kamae' / 'explain shizen no kamae stance'.
    """
    records = _load_kamae_records()
    if not records:
        return None

    q = _fold(question)
    for key, rec in records.items():
        if not key:
            continue
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, q):
            return _format_kamae(rec)

    return None

# ----------------- rank-based kamae (nttv rank requirements.txt) -----------------

def _load_rank_text() -> str:
    return _load_file("nttv rank requirements.txt")


def _extract_rank_kamae(rank_label: str) -> Optional[List[str]]:
    """
    Find the 'Kamae:' line for a given rank block, e.g. '9th Kyu'.

    We:
      - Locate the line that matches the rank label.
      - Scan forward until the next blank line or next 'Kyu' header.
      - Within that window, pick the first line that starts with 'Kamae:' (not 'Weapon Kamae:').
    """
    text = _load_rank_text()
    if not text:
        return None

    lines = text.splitlines()
    # Find the rank header line index
    start_idx = None
    target = _fold(rank_label)
    for i, raw in enumerate(lines):
        if _fold(raw.strip()) == target:
            start_idx = i
            break

    if start_idx is None:
        return None

    # Scan forward for the first "Kamae:" line in this rank block
    kamae_line: Optional[str] = None
    for raw in lines[start_idx + 1:]:
        stripped = raw.strip()
        if not stripped:
            # end of this rank block
            break
        # next rank header?
        if re.search(r"\b(\d+)(st|nd|rd|th)\s+kyu\b", _fold(stripped)):
            break
        # Must start with 'Kamae:' exactly, not 'Weapon Kamae:'
        if stripped.startswith("Kamae:"):
            kamae_line = stripped
            break

    if kamae_line is None:
        return None

    # Take everything after "Kamae:"
    after = kamae_line.split(":", 1)[1].strip()
    if not after:
        return []
    parts = [p.strip() for p in after.split(";") if p.strip()]
    return parts


def _answer_rank_kamae(question: str) -> Optional[str]:
    """
    Handle questions like 'what are the kamae for 9th kyu?'
    """
    q = _fold(question)
    m = re.search(r"(\d+)(st|nd|rd|th)\s+kyu", q)
    if not m:
        return None

    num = m.group(1)
    suffix = m.group(2)
    label = f"{num}{suffix} Kyu"  # 9th Kyu, 8th Kyu, 1st Kyu, etc.

    kamae_list = _extract_rank_kamae(label)
    if kamae_list is None:
        return None
    if not kamae_list:
        return f"No specific kamae are listed for {label}."

    joined = join_oxford(kamae_list)
    return f"Kamae for {label}: {joined}"

# ----------------- weapon kamae (NTTV Weapons Reference.txt) -----------------

def _load_weapons_text() -> str:
    return _load_file("NTTV Weapons Reference.txt")


def _build_weapon_kamae_index() -> Dict[str, List[str]]:
    """
    Build alias -> [kamae, ...] mapping from NTTV Weapons Reference.txt.
    """
    text = _load_weapons_text()
    if not text:
        return {}

    index: Dict[str, List[str]] = {}
    current_weapon: Optional[str] = None
    current_aliases: List[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        if line.startswith("[WEAPON]"):
            name = line[len("[WEAPON]"):].strip()
            current_weapon = name
            current_aliases = [name]

        elif line.upper().startswith("ALIASES:"):
            alias_str = line.split(":", 1)[1]
            aliases = [a.strip() for a in alias_str.split(",") if a.strip()]
            current_aliases.extend(aliases)

        elif line.upper().startswith("KAMAE:"):
            if not current_weapon:
                continue
            kamae_str = line.split(":", 1)[1]
            kamae = [k.strip() for k in kamae_str.split(",") if k.strip()]

            for al in current_aliases:
                index[_fold(al)] = kamae

    return index


def _answer_weapon_kamae(question: str) -> Optional[str]:
    """
    Handle questions like:
      - 'What kamae do we use with the hanbo?'
      - 'Hanbo kamae'
      - 'weapon kamae for rokushakubo'
    """
    idx = _build_weapon_kamae_index()
    if not idx:
        return None

    q = _fold(question)
    best_alias = None
    for alias in idx.keys():
        if alias and alias in q:
            best_alias = alias
            break

    if not best_alias:
        return None

    kamae = idx[best_alias]
    display_weapon = best_alias.title()
    joined = join_oxford(kamae)
    return f"{display_weapon} kamae: {joined}"

# ----------------- public entrypoint -----------------

def try_answer_kamae(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic Kamae extractor:

      * Rank-based lists: 'kamae for 9th kyu'
      * Weapon kamae: 'hanbo kamae', 'kamae with the rokushakubo'
      * Specific kamae definitions: 'what is Hicho no Kamae'
    """
    if not _looks_like_kamae_question(question):
        return None

    # 1) Rank-specific kamae (e.g., "kamae for 9th kyu")
    ans = _answer_rank_kamae(question)
    if ans:
        return ans

    # 2) Weapon kamae (e.g., "Hanbo kamae")
    ans = _answer_weapon_kamae(question)
    if ans:
        return ans

    # 3) Individual kamae definitions (e.g., "what is Hicho no Kamae")
    ans = _answer_specific_kamae(question)
    if ans:
        return ans

    return None
