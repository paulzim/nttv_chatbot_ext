# extractors/kihon_happo.py
from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple
from .common import dedupe_preserve, join_oxford

TRIG = re.compile(r"\b(kihon\s*happ[oō]|kihon|happ[oō])\b", re.I)

# subset headings and aliases
KOSHI_ALIASES = [
    r"kosshi\s+kohon\s+sanpo",     # common OCR / typo variants covered below
    r"kosshi\s+kihon\s+sanpo",
    r"koshi\s+kihon\s+sanpo",
    r"kosshi\s+sanpo",
    r"koshi\s+sanpo",
]
TORITE_ALIASES = [
    r"torite\s+goho(?:\s+gata)?",
    r"torite-goho(?:\s+gata)?",
    r"torite\s+gohō(?:\s+gata)?",
]

# tolerant line patterns
SEP = re.compile(r"[;,]")  # list items separated by semicolons/commas
BULLET = re.compile(r"^\s*[-•]\s*(.+?)\s*$")
HEADING = re.compile(r"^\s*([A-Z][A-Za-z0-9'’\-\s]+?)\s*:\s*$")

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def _split_items(line: str) -> List[str]:
    parts = [p.strip(" -•\t") for p in SEP.split(line) if p.strip()]
    # filter out obviously long sentences
    return [p for p in parts if 2 <= len(p) <= 60]

def _find_first_definition(text: str) -> str | None:
    for raw in text.splitlines():
        line = _norm(raw)
        if not line:
            continue
        if re.search(r"\bkihon\s*happ[oō]\b", line, re.I) and 12 <= len(line) <= 220:
            return line.rstrip(" ;,")
    return None

def _harvest_lists(text: str) -> List[str]:
    items: List[str] = []
    for raw in text.splitlines():
        line = _norm(raw)
        if not line:
            continue
        # lines that look listy
        if SEP.search(line) and len(line) < 240:
            items.extend(_split_items(line))
        else:
            m = BULLET.match(line)
            if m:
                items.append(_norm(m.group(1)))
    return dedupe_preserve(items)

def _alias_pat(aliases: List[str]) -> re.Pattern:
    return re.compile(r"(?i)\b(" + "|".join(aliases) + r")\b")

KOSHI_PAT = _alias_pat(KOSHI_ALIASES)
TORITE_PAT = _alias_pat(TORITE_ALIASES)

def _collect_subset(text: str, start_idx: int) -> Tuple[List[str], int]:
    """
    Starting at a heading line, collect following bullet/list items until blank or next heading.
    Returns (items, next_index).
    """
    lines = text.splitlines()
    items: List[str] = []
    i = start_idx + 1
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            break
        if HEADING.match(ln):
            break
        if SEP.search(ln):
            items.extend(_split_items(ln))
        else:
            m = BULLET.match(ln)
            if m:
                items.append(_norm(m.group(1)))
        i += 1
    return dedupe_preserve(items), i

def _group_by_subsets(text: str) -> Tuple[List[str], List[str]]:
    """
    Try to find the two classic subsets and their items:
      - Kosshi Kihon Sanpo (usually 3 items)
      - Torite Goho (usually 5 items)
    """
    koshi: List[str] = []
    torite: List[str] = []

    lines = text.splitlines()
    for idx, raw in enumerate(lines):
        ln = _norm(raw)
        if not ln:
            continue

        # Heading forms like "Kosshi Kihon Sanpo:" then bullets/lines
        if KOSHI_PAT.search(ln) and (ln.endswith(":") or ln.endswith("—") or ln.endswith("-")):
            items, _ = _collect_subset(text, idx)
            if items:
                koshi.extend(items)
            continue

        if TORITE_PAT.search(ln) and (ln.endswith(":") or ln.endswith("—") or ln.endswith("-")):
            items, _ = _collect_subset(text, idx)
            if items:
                torite.extend(items)
            continue

        # Inline forms like "Kosshi Kihon Sanpo: a, b, c"
        if KOSHI_PAT.search(ln) and ":" in ln:
            after = ln.split(":", 1)[1].strip()
            koshi.extend(_split_items(after))
            continue

        if TORITE_PAT.search(ln) and ":" in ln:
            after = ln.split(":", 1)[1].strip()
            torite.extend(_split_items(after))
            continue

    return dedupe_preserve(koshi), dedupe_preserve(torite)

def try_answer_kihon_happo(question: str, passages: List[Dict[str, Any]]) -> str | None:
    if not TRIG.search(question):
        return None

    defs: List[str] = []
    flat_items: List[str] = []
    koshi_all: List[str] = []
    torite_all: List[str] = []

    for p in passages[:8]:
        text = p.get("text", "")
        if not text or len(text) < 20:
            continue

        # collect a concise definition if present
        d = _find_first_definition(text)
        if d:
            defs.append(d)

        # collect any flat list-y items we can spot
        flat_items.extend(_harvest_lists(text))

        # try to group by the two subsets
        k, t = _group_by_subsets(text)
        if k:
            koshi_all.extend(k)
        if t:
            torite_all.extend(t)

    koshi_all = dedupe_preserve(koshi_all)
    torite_all = dedupe_preserve(torite_all)
    flat_items = dedupe_preserve(flat_items)

    # 1) Best case: we found the two subsets — return grouped, readable, 1–2 sentences.
    if koshi_all or torite_all:
        parts: List[str] = ["Kihon Happo consists of Kosshi Kihon Sanpo and Torite Goho."]
        # Add subset details when we have them
        if koshi_all:
            parts.append(f"Kosshi Kihon Sanpo: {join_oxford(koshi_all[:3])}.")
        if torite_all:
            parts.append(f"Torite Goho: {join_oxford(torite_all[:5])}.")
        return " ".join(parts)

    # 2) Next: if we harvested ~8 items total, give a compact 8-name list.
    if 6 <= len(flat_items) <= 12:
        return f"Kihon Happo: {join_oxford(flat_items[:8])}."

    # 3) Fallback: a concise definition sentence if available.
    if defs:
        sent = re.split(r"(?<=[.!?])\s+", defs[0])[0]
        return sent

    # No deterministic hit; let RAG handle it.
    return None
