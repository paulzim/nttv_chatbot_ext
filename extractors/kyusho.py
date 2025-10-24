# extractors/kyusho.py
# Deterministic Kyusho extractor
# - Triggers on "kyusho"/"pressure point" or when a known point name appears in the question
# - Parses entries from retrieved passages only (no filesystem access)
# - Returns either a concise list of points or a one-line location/description for a specific point

from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple
from .common import dedupe_preserve, join_oxford

# ----------------------------
# Parsing helpers (tolerant)
# ----------------------------

# Matches "Name: description" or "Name – description" styles
LINE_KV = re.compile(r"^\s*[-•]?\s*([A-Z][A-Za-z0-9'’\-\s]+?)\s*[:–-]\s*(.+?)\s*$")

# Matches "heading:" followed by bullet lines we can harvest
HEADING = re.compile(r"^\s*([A-Z][A-Za-z0-9'’\-\s]+?)\s*:\s*$")
BULLET  = re.compile(r"^\s*[-•]\s*(.+?)\s*$")

# Generic “listy” line that might contain multiple points with commas/semicolons
LISTY  = re.compile(r"[;,]")

# Simple “where/what” detectors
ASK_LIST = re.compile(r"\b(list|what\s+are|which)\b.*\b(kyusho|pressure\s*points?)\b", re.I)
ASK_WHERE = re.compile(r"\b(where\s+is|location\s+of|where\s+are)\b", re.I)
ASK_WHAT  = re.compile(r"\b(what\s+is|describe|definition\s+of)\b", re.I)

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def _looks_like_point_name(s: str) -> bool:
    # Heuristic: short-ish proper names, allow ryu-like tokens but not sentences
    # Many kyusho names are 1–3 words, often capitalized or romanized
    w = s.strip()
    return 2 <= len(w) <= 40 and not w.endswith((".", ":", ";"))

def _split_candidates(line: str) -> List[str]:
    # Break a semicolon/comma-heavy line into name-like tokens
    parts = [p.strip(" -•\t") for p in re.split(r"[;,]", line) if p.strip()]
    # Keep only plausible “names”
    return [p for p in parts if _looks_like_point_name(p)]

def _ingest_line_kv(line: str, store: Dict[str, str]) -> bool:
    m = LINE_KV.match(line)
    if not m:
        return False
    name, desc = _norm(m.group(1)), _norm(m.group(2))
    if len(name) < 2 or len(desc) < 2:
        return False
    # Prefer first occurrence; later duplicates are ignored to stay stable
    store.setdefault(name, desc)
    return True

def _parse_kyusho_entries(text: str) -> Dict[str, str]:
    """Parse kyusho name → short description/location from a block of text."""
    entries: Dict[str, str] = {}
    lines = text.splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        # 1) "Name: desc" direct form
        if _ingest_line_kv(line, entries):
            i += 1
            continue

        # 2) "Heading:" followed by bullets
        mh = HEADING.match(line)
        if mh:
            # Accumulate bullets until blank or next heading
            j = i + 1
            bucket: List[str] = []
            while j < len(lines) and lines[j].strip():
                mb = BULLET.match(lines[j])
                if mb:
                    bucket.append(_norm(mb.group(1)))
                    j += 1
                else:
                    break
            # Convert bullets into entries if they look like "Name - desc" or "Name: desc"
            for b in bucket:
                if _ingest_line_kv(b, entries):
                    continue
                # Fallback: treat as a candidate name with no explicit desc
                # (we store empty desc; later we can still answer list-type queries)
                if _looks_like_point_name(b):
                    entries.setdefault(b, "")
            i = j
            continue

        # 3) Listy lines with multiple candidates (e.g., “X, Y, Z (temple), ...”)
        if LISTY.search(line):
            for cand in _split_candidates(line):
                entries.setdefault(cand, "")
            i += 1
            continue

        i += 1

    return entries

def _gather_entries(passages: List[Dict[str, Any]], limit: int = 8) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for p in passages[:limit]:
        text = p.get("text", "")
        if not text or len(text) < 10:
            continue
        block = _parse_kyusho_entries(text)
        for k, v in block.items():
            if k not in merged or (not merged[k] and v):
                merged[k] = v
    return merged

def _match_point_in_question(question: str, names: List[str]) -> Tuple[str | None, List[str]]:
    ql = question.lower()
    hits = []
    for n in names:
        nl = n.lower()
        if nl in ql:
            hits.append(n)
    best = hits[0] if hits else None
    return best, hits

# ----------------------------
# Public extractor entry
# ----------------------------

def try_answer_kyusho(question: str, passages: List[Dict[str, Any]]) -> str | None:
    """
    Returns:
      - A short comma list of kyusho names (when user asks to list)
      - OR a one-sentence location/description for a specific point
      - OR None (let other extractors / RAG handle it)
    """
    ql = question.lower()
    trigger_words = ("kyusho", "kyūsho", "pressure point", "pressure points")
    if not any(w in ql for w in trigger_words):
        # We also allow implicit triggers if the question contains a known point name
        # after we parse entries below.
        pass

    entries = _gather_entries(passages, limit=8)
    if not entries:
        return None

    # If no trigger word, see if the question mentions a known point explicitly
    if not any(w in ql for w in trigger_words):
        name_hit, _ = _match_point_in_question(question, list(entries.keys()))
        if not name_hit:
            return None

    # Handle "where is", "location of", "what is" types first
    name_hit, _ = _match_point_in_question(question, list(entries.keys()))
    if name_hit:
        desc = entries.get(name_hit, "").strip()
        if desc:
            # Keep it to one clean sentence; strip trailing punctuation noise
            return f"{name_hit}: {desc.rstrip(' ;,')}"
        else:
            # We know the point name, but no description present in context
            return f"{name_hit}: (location/description not listed in the provided context)."

    # Otherwise assume list request if explicitly asked
    if ASK_LIST.search(question) or "kyusho" in ql or "pressure point" in ql:
        names = dedupe_preserve([k for k in entries.keys()])
        if not names:
            return None
        # Short, readable comma list; cap at ~20 to avoid walls of text
        names = names[:20]
        return join_oxford(names)

    return None
