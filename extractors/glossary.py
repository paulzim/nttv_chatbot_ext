# extractors/glossary.py
"""Deterministic glossary fallback for single-term definition questions.

This is intentionally conservative and only kicks in when:
- The question clearly looks like a "what is / define / meaning of" style query, OR
- The question is very short (1–3 words) that looks like a term by itself.

It reads from:
- Retrieved passages whose source is "Glossary - edit.txt", and/or
- The on-disk Glossary file if present (../data/Glossary - edit.txt).

It also performs a disambiguation step:
- If the query clearly looks like a technique/kata name, the glossary backs off
  and returns None so technique / rank extractors (or the LLM) can answer.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional


# ----------------------------------------------------------------------
# Basic helpers
# ----------------------------------------------------------------------

def _fold(s: str) -> str:
    """Lowercase, collapse whitespace, strip basic punctuation."""
    s = (s or "")
    s = s.replace("\u2010", "-").replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("–", "-").replace("—", "-")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _same_source_name(p_source: str, target_name: str) -> bool:
    """Basename + lowercase match for passage.source vs logical filename."""
    if not p_source:
        return False
    base_actual = os.path.basename(p_source).lower()
    base_target = os.path.basename(target_name).lower()
    return base_actual == base_target


def _looks_like_glossary_question(question: str) -> bool:
    q = _fold(question)
    # Strong definition signals
    if any(t in q for t in ["what is", "what's", "define", "definition of", "meaning of", "what does", "translate"]):
        return True
    # Very short term-like query: e.g., "Happo Geri" or "Hicho no Kamae"
    tokens = q.split()
    if 1 <= len(tokens) <= 3 and not any(t in q for t in ["who", "when", "where", "why", "how"]):
        return True
    return False


# ----------------------------------------------------------------------
# Glossary loading & parsing
# ----------------------------------------------------------------------

def _load_full_glossary_file() -> str:
    """Try to read the full Glossary file from disk (if available)."""
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "data" / "Glossary - edit.txt",
        here.parent / "Glossary - edit.txt",
    ]
    for p in candidates:
        try:
            if p.exists():
                return p.read_text(encoding="utf-8")
        except Exception:
            continue
    return ""


def _gather_glossary_text(passages: List[Dict[str, Any]]) -> str:
    """Combine any retrieved glossary chunks with the full file on disk."""
    chunks: List[str] = []
    for p in passages:
        src_raw = p.get("source") or ""
        if _same_source_name(src_raw, "Glossary - edit.txt") or "glossary" in src_raw.lower():
            txt = p.get("text", "")
            if txt:
                chunks.append(txt)

    full_file = _load_full_glossary_file()
    if full_file.strip():
        chunks.append(full_file)

    return "\n".join(chunks)


def _parse_glossary(text: str) -> Dict[str, tuple[str, str]]:
    """Parse lines of the form 'Term - Definition' into a mapping.

    Returns: { folded_term -> (display_term, definition) }
    Handles simple continuation lines (lines that don't contain a dash) as
    extensions of the previous definition.
    """
    entries: Dict[str, tuple[str, str]] = {}
    last_key: Optional[str] = None

    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        # Skip a top-level header like 'Glossary'
        if line.lower() == "glossary":
            continue

        # Match "Term - Definition" with various dash types
        m = re.match(r"^(.+?)\s*[-–—]\s*(.+)$", line)
        if m:
            term = m.group(1).strip()
            definition = m.group(2).strip()
            key = _fold(term)
            if key and key not in entries:
                entries[key] = (term, definition)
                last_key = key
            else:
                last_key = key if key in entries else None
            continue

        # Continuation of previous definition (no dash on this line)
        if last_key is not None and line:
            term, definition = entries[last_key]
            definition = f"{definition} {line}".strip()
            entries[last_key] = (term, definition)

    return entries


def _extract_candidate(question: str) -> str:
    """Extract the term fragment from a 'what is / define / meaning of' question."""
    q = question.strip()
    m = re.search(
        r"(?:what\s+is|what's|define|definition of|meaning of|what does)\s+(.+)$",
        q,
        flags=re.IGNORECASE,
    )
    cand = (m.group(1) if m else q).strip()
    # Remove trailing question/punctuation
    cand = cand.rstrip("?!., ")
    # Strip common noise suffixes
    cand = re.sub(
        r"\b(in japanese|in ninjutsu|in bujinkan|term|word|mean|meaning)\b",
        "",
        cand,
        flags=re.IGNORECASE,
    )
    return cand.strip()


def _choose_glossary_entry(
    question: str, entries: Dict[str, tuple[str, str]]
) -> Optional[tuple[str, str]]:
    if not entries:
        return None

    cand_raw = _extract_candidate(question)
    cand_fold = _fold(cand_raw)
    if not cand_fold or len(cand_fold) < 3:
        return None

    # 1) Direct key match
    if cand_fold in entries:
        return entries[cand_fold]

    # 2) Allow substring containment for reasonably specific terms
    for key, (term, definition) in entries.items():
        if len(key) >= 4 and (cand_fold in key or key in cand_fold):
            return term, definition

    # 3) Fallback: try using just the last 1–2 words as the term
    tokens = cand_fold.split()
    for span in (2, 1):
        if len(tokens) >= span:
            sub = " ".join(tokens[-span:])
            for key, (term, definition) in entries.items():
                if sub == key or (len(sub) >= 4 and (sub in key or key in sub)):
                    return term, definition

    return None


# ----------------------------------------------------------------------
# Technique / kata disambiguation
# ----------------------------------------------------------------------

_TECH_NAME_HINTS = [
    "gyaku", "kudaki", "dori", "gatame", "otoshi", "nage",
    "seoi", "kote", "musha", "juji", "kaiten", "gake", "ori",
]


def _gather_technique_text(passages: List[Dict[str, Any]]) -> str:
    """Collect Technique Descriptions text from retrieved passages."""
    chunks: List[str] = []
    for p in passages:
        src_raw = p.get("source") or ""
        if _same_source_name(src_raw, "Technique Descriptions.md") or "technique descriptions" in src_raw.lower():
            txt = p.get("text", "")
            if txt:
                chunks.append(txt)
    return "\n".join(chunks)


def _extract_technique_names(md_text: str) -> set[str]:
    """Extract technique names (first CSV column) from Technique Descriptions.md."""
    names: set[str] = set()
    for raw in (md_text or "").splitlines():
        st = raw.strip()
        if not st:
            continue
        if st.startswith("#") or st.startswith("```"):
            continue
        if "," not in raw:
            continue
        first = raw.split(",", 1)[0].strip()
        if first:
            names.add(_fold(first))
    return names


def _looks_like_technique_term(question: str, passages: List[Dict[str, Any]]) -> bool:
    """
    Return True if the question appears to be about a specific technique/kata.

    This has two layers:
    1) Pure heuristic: words like 'kudaki', 'gyaku', 'dori', 'nage', 'kata', 'waza'
       strongly suggest a technique/kata and are enough to make the glossary back off.
    2) Data-aware: if Technique Descriptions are present and the candidate matches
       a known technique name, also back off.
    """
    q = _fold(question)

    # Strong kata/waza indicators
    if " no kata" in q or "kata" in q or "waza" in q:
        return True

    # Name-pattern hints (Oni Kudaki, Omote Gyaku, Musha Dori, etc.)
    if any(h in q for h in _TECH_NAME_HINTS):
        return True

    # Data-aware check using Technique Descriptions (if available)
    cand = _fold(_extract_candidate(question))
    if len(cand) < 3:
        return False

    md_text = _gather_technique_text(passages)
    if not md_text.strip():
        return False

    names = _extract_technique_names(md_text)
    if not names:
        return False

    # Exact match to a technique name
    if cand in names:
        return True

    # Try last 1–2 words (e.g., "oni kudaki", "koku")
    tokens = cand.split()
    for span in (2, 1):
        if len(tokens) >= span:
            sub = " ".join(tokens[-span:])
            if sub in names:
                return True

    return False


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def try_answer_glossary(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """Glossary-based fallback definition for single-term style questions."""
    if not _looks_like_glossary_question(question):
        return None

    # Disambiguation: if this clearly looks like a technique/kata, let the
    # technique/rank extractors (or LLM) answer instead.
    if _looks_like_technique_term(question, passages):
        return None

    text = _gather_glossary_text(passages)
    if not text.strip():
        return None

    entries = _parse_glossary(text)
    term_def = _choose_glossary_entry(question, entries)
    if not term_def:
        return None

    term, definition = term_def
    return f"{term}: {definition}"
