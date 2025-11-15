# extractors/weapons.py
"""
Deterministic weapons extractor.

This module parses **NTTV Weapons Reference.txt** style blocks and can answer:
- Weapon profiles: “What is the hanbo weapon?”, “Explain the kusari fundo weapon”
- Weapon rank: “At what rank do I learn kusari fundo?”
- Katana terminology: “what are the parts of the katana?”
- Shuriken classification: “what are the types of shuriken?”
"""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional


# ----------------------------
# Static texts for special cases
# ----------------------------

KATANA_PARTS_TEXT = (
    "Parts of the katana:\n"
    "- Tsuka – handle\n"
    "- Tsuka Kishiri – handle endcap\n"
    "- Saya – sheath\n"
    "- Sageo – cord for the sheath\n"
    "- Tsuba – handguard\n"
    "- Ha – blade edge\n"
    "- Hi – blood gutter\n"
    "- Hamon – temper line\n"
    "- Mune – back of the sword\n"
    "- Kissaki – the tip or point area that has a ridgeline."
)

SHURIKEN_TYPES_TEXT = (
    "Types of shuriken:\n"
    "\n"
    "1) Bō-shuriken (spike-type)\n"
    "   - Straight spike, like a needle, dart, or large nail.\n"
    "   - Usually steel or iron, often 12–18 cm long.\n"
    "   - Commonly thrown with a single spin from the center or tail.\n"
    "\n"
    "2) Hira-shuriken (flat “star” type)\n"
    "   - Flat, bladed forms – the classic ninja “throwing star”.\n"
    "   - Typically 4–8 points, edges sharpened.\n"
    "   - Thrown with a slicing rotation at short range.\n"
    "\n"
    "3) Senban-shuriken\n"
    "   - Square, four-point subtype of hira-shuriken.\n"
    "   - Can be stuck in the ground as a trap or thrown edge-first.\n"
    "\n"
    "4) Needle / hari-gata shuriken\n"
    "   - Very thin needle-like forms.\n"
    "   - Good for precise, stealthy insertion and often historically linked to poison use.\n"
    "\n"
    "5) Modern improvised throwing tools\n"
    "   - Chopsticks, pens, nails, hex keys, etc.\n"
    "   - If it fits in the hand, can be controlled, and can pierce or distract, "
    "you can train it as shuriken."
)


# ----------------------------
# Helper utilities
# ----------------------------

def _norm(text: str) -> str:
    return text.lower().strip()


def _join_passages_text(passages: List[Dict[str, Any]], source_name: str) -> str:
    """Concatenate text from all passages matching a given source."""
    chunks: List[str] = []
    for p in passages:
        if _norm(p.get("source", "")) == _norm(source_name):
            t = p.get("text") or ""
            chunks.append(t)
    return "\n".join(chunks)


def _parse_weapon_blocks(passages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Parse [WEAPON] blocks from NTTV Weapons Reference text into dictionaries.

    Each block looks like:

        [WEAPON] Hanbo
        ALIASES: ...
        TYPE: ...
        KAMAE: ...
        CORE ACTIONS: ...
        RANKS: Introduced at 8th Kyu
        NOTES: ...

    We keep keys upper-cased: NAME, ALIASES, TYPE, KAMAE, CORE ACTIONS, RANKS, NOTES, RAW.
    """
    text = _join_passages_text(passages, "NTTV Weapons Reference.txt")
    if not text:
        return []

    rows: List[Dict[str, str]] = []
    # Split on new weapon headers
    blocks = re.split(r"(?m)^(?=\[WEAPON\] )", text)
    for block in blocks:
        block = block.strip()
        if not block or not block.startswith("[WEAPON]"):
            continue

        data: Dict[str, str] = {"RAW": block}
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("[WEAPON]"):
                name = line[len("[WEAPON]") :].strip()
                data["NAME"] = name
            elif ":" in line:
                key, val = line.split(":", 1)
                data[key.strip().upper()] = val.strip()
        if "NAME" in data:
            rows.append(data)
    return rows


def _aliases_for_row(row: Dict[str, str]) -> List[str]:
    aliases: List[str] = []
    name = row.get("NAME", "")
    if name:
        aliases.append(name.lower())
    alias_str = row.get("ALIASES", "")
    if alias_str:
        for part in alias_str.split(","):
            part = part.strip()
            if part:
                aliases.append(part.lower())
    # de-duplicate
    seen = set()
    result: List[str] = []
    for a in aliases:
        if a not in seen:
            seen.add(a)
            result.append(a)
    return result


def _find_weapon_row(question: str, rows: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    q = _norm(question)
    for row in rows:
        for alias in _aliases_for_row(row):
            if alias and alias in q:
                return row
    return None


# ----------------------------
# Public extractor functions
# ----------------------------

def try_answer_katana_parts(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """Answer questions about the parts / terminology of the katana."""
    q = _norm(question)
    if "katana" not in q:
        return None
    if "part" in q or "terminology" in q or "terms" in q or "name" in q:
        return KATANA_PARTS_TEXT
    return None


def try_answer_weapon_profile(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Return a structured weapon profile for questions like:
    - "What is the hanbo weapon?"
    - "Explain the kusari fundo weapon."
    - "Tell me about the katana."
    """
    q = _norm(question)

    # Special case: shuriken classification (types of shuriken)
    if "shuriken" in q and ("type" in q or "types" in q or "kind" in q or "kinds" in q):
        return SHURIKEN_TYPES_TEXT

    rows = _parse_weapon_blocks(passages)
    if not rows:
        return None

    row = _find_weapon_row(question, rows)
    if not row:
        return None

    name = row.get("NAME", "Weapon").strip()
    typ = row.get("TYPE", "").strip()
    kamae = row.get("KAMAE", "").strip()
    core = row.get("CORE ACTIONS", "").strip()
    ranks = row.get("RANKS", "").strip()
    notes = row.get("NOTES", "").strip()

    parts: List[str] = []
    parts.append(f"{name} weapon profile:")
    if typ:
        parts.append(f"Type: {typ}.")
    if kamae:
        parts.append(f"Kamae: {kamae}.")
    if core:
        # Tests expect the exact phrase "core actions include"
        parts.append(f"Core actions include: {core}.")
    if ranks:
        parts.append(f"Ranks: {ranks}.")
    if notes:
        parts.append(f"Notes: {notes}.")

    return " ".join(parts)


def try_answer_weapon_rank(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Answer questions like:
    - "At what rank do I learn kusari fundo?"
    - "When is hanbo introduced?"
    """
    q = _norm(question)
    if "rank" not in q and "kyu" not in q and "introduced" not in q:
        return None

    rows = _parse_weapon_blocks(passages)
    if not rows:
        return None

    row = _find_weapon_row(question, rows)
    if not row:
        return None

    name = row.get("NAME", "This weapon").strip()
    ranks = row.get("RANKS", "").strip()
    if not ranks:
        return None

    # RANKS line usually looks like "Introduced at 4th Kyu".
    low_ranks = ranks.lower()
    if low_ranks.startswith("introduced at "):
        pretty = ranks[len("Introduced at ") :].strip()
    else:
        pretty = ranks

    return f"You first study {name} at {pretty}."
