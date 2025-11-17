from __future__ import annotations
from typing import List, Dict, Any, Optional
import re
import os

# ============================================================
# Small, safe helpers (keep behavior stable)
# ============================================================

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _lc(s: str) -> str:
    return _norm(s).lower()

def _join_human(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f", {items[-1]}"

def _dedup(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for s in seq:
        s_n = _norm(s)
        k = s_n.lower()
        if s_n and k not in seen:
            out.append(s_n)
            seen.add(k)
    return out

def _same_source_name(p_source: str, target_name: str) -> bool:
    """
    Compare FAISS/meta 'source' values (which may include paths) with the
    logical filename used by the extractor. Basename + lowercase.
    """
    if not p_source:
        return False
    base_actual = os.path.basename(p_source).lower()
    base_target = os.path.basename(target_name).lower()
    return base_actual == base_target

# ============================================================
# Alias tables for nicer display (kicks + punches)
# ============================================================

_KICK_ALIASES: Dict[str, List[str]] = {
    "zenpo geri": ["Mae Geri", "Front Kick"],
    "mae geri": ["Zenpo Geri", "Front Kick"],
    "front kick": ["Zenpo Geri", "Mae Geri"],
    "sokuho geri": ["Side Kick"],
    "koho geri": ["Back Kick"],
}

_PUNCH_ALIASES: Dict[str, List[str]] = {
    "fudo ken": ["Immovable Fist"],
    "shuto": ["Knife-hand"],
    "shuto uchi": ["Knife-hand Strike"],
    "shikan ken": ["Foreknuckle Fist"],
    "shako ken": ["Claw Hand"],
    "boshi ken": ["Thumb Knuckle Strike"],
    "tsuki": ["Punch"],
    "jodan tsuki": ["High Punch"],
    "gedan tsuki": ["Low Punch"],
    "ken kudaki": ["Fist Crusher"],
    "happa ken": ["Double-Palm Strike"],
    "kikaku ken": ["Headbutt"],
}

def _with_kick_aliases(name: str) -> str:
    key = _lc(name)
    aliases = _KICK_ALIASES.get(key, [])
    if not aliases:
        return _norm(name)
    seen = {_norm(name)}
    alias_clean = []
    for a in aliases:
        a_n = _norm(a)
        if a_n not in seen:
            alias_clean.append(a_n)
            seen.add(a_n)
    return f"{_norm(name)} ({' / '.join(alias_clean)})" if alias_clean else _norm(name)

def _with_punch_aliases(name: str) -> str:
    key = _lc(name)
    aliases = _PUNCH_ALIASES.get(key, [])
    if not aliases:
        return _norm(name)
    seen = {_norm(name)}
    alias_clean = []
    for a in aliases:
        a_n = _norm(a)
        if a_n not in seen:
            alias_clean.append(a_n)
            seen.add(a_n)
    return f"{_norm(name)} ({' / '.join(alias_clean)})" if alias_clean else _norm(name)

# ============================================================
# Rank parsing
# ============================================================

_RANK_HEADER_RE = re.compile(
    r"^(?P<hdr>(?:\d+(?:st|nd|rd|th)\s+kyu|shodan))\b",
    re.IGNORECASE | re.MULTILINE
)

def _rank_key_from_question(q: str) -> Optional[str]:
    ql = _lc(q)
    m = re.search(r"\b(\d+)\s*(?:st|nd|rd|th)?\s*kyu\b", ql)
    if m:
        n = m.group(1)
        if n == "1":
            return "1st kyu"
        if n == "2":
            return "2nd kyu"
        if n == "3":
            return "3rd kyu"
        return f"{n}th kyu"
    if "shodan" in ql:
        return "shodan"
    return None

def _find_rank_text_from_passages(passages: List[Dict[str, Any]]) -> Optional[str]:
    # Prefer explicitly injected rank requirements
    for p in passages:
        src = (p.get("source") or p.get("meta", {}).get("source") or "")
        text = p.get("text", "")
        if text and (_same_source_name(src, "nttv rank requirements.txt") or "nttv rank requirements" in src.lower()):
            return text
    # Fallback: any chunk that clearly looks like a rank document
    for p in passages:
        text = (p.get("text") or "")
        if text and ("kyu" in text.lower() and "kamae" in text.lower()):
            return text
    return None

def _extract_rank_block(full_text: str, rank_key: str) -> Optional[str]:
    if not full_text or not rank_key:
        return None
    pattern = re.compile(rf"^(?P<hdr>{re.escape(rank_key)})\b.*$", re.IGNORECASE | re.MULTILINE)
    start_m = pattern.search(full_text)
    if not start_m:
        return None
    start = start_m.start()
    next_m = _RANK_HEADER_RE.search(full_text, pos=start + 1)
    end = next_m.start() if next_m else len(full_text)
    return full_text[start:end].strip()

def _extract_section_lines(block: str, header_label: str) -> List[str]:
    """
    Get lines for a header like "Striking:".
    IMPORTANT FIX: capture items that appear on the SAME LINE as the header,
    e.g., "Striking: Fudo Ken; ...".
    """
    if not block:
        return []

    # Find the header line and capture optional inline content after ':' on that same line
    hdr_line_re = re.compile(
        rf"^(?P<header>\s*{re.escape(header_label)})\s*(?P<inline>.*)$",
        re.IGNORECASE | re.MULTILINE
    )
    m = hdr_line_re.search(block)
    if not m:
        return []

    # Inline items present after the header on the same line?
    inline = m.group("inline").strip()
    # Slice the text AFTER the header line
    tail = block[m.end():]

    # Stop at next section header (line ending with ':') OR next rank header
    stop = len(tail)
    next_section = re.search(r"^[A-Za-z0-9].*?:\s*$", tail, re.MULTILINE)
    if next_section:
        stop = min(stop, next_section.start())
    next_rank = _RANK_HEADER_RE.search(tail)
    if next_rank:
        stop = min(stop, next_rank.start())

    body = tail[:stop]

    out: List[str] = []
    if inline:
        out.append(inline)  # keep inline content as the first “line”
    # Then add subsequent non-empty lines
    out.extend(ln.strip() for ln in body.splitlines() if _norm(ln))
    return out

def _split_items(lines: List[str]) -> List[str]:
    items = []
    for ln in lines:
        parts = [x.strip(" -•\t") for x in re.split(r"[;,]", ln) if x and len(x.strip()) > 1]
        items.extend(parts)
    return [i for i in (_norm(x) for x in items) if i]

# ============================================================
# Public extractors
# ============================================================

def try_answer_rank_striking(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Answer kick/punch lists for a specific rank.
    Default: rank-only (what is assessed/introduced at that rank).
    If user asks cumulative (e.g., "need to know by 8th kyu"), merge 9th-kyu foundational kicks.
    """
    ql = _lc(question)
    # intent
    wants_kicks = any(w in ql for w in ["kick", "kicks", "geri"])
    wants_punches = any(w in ql for w in ["punch", "punches", "tsuki", "ken", "strike", "striking"])
    if not (wants_kicks or wants_punches):
        return None

    # cumulative intent? (BROADER)
    cumulative = (
        ("need to know" in ql) or
        any(phrase in ql for phrase in [
            "need to know by", "up through", "up to", "all kicks for", "everything for", "study list"
        ]) or
        re.search(r"\bby\s+\d+(st|nd|rd|th)\s+kyu\b", ql) is not None
    )

    rank_key = _rank_key_from_question(question)
    if not rank_key:
        return None

    rank_text = _find_rank_text_from_passages(passages)
    if not rank_text:
        return None

    # Current-rank block
    block = _extract_rank_block(rank_text, rank_key)
    if not block:
        return None

    lines = _extract_section_lines(block, "Striking:")
    if not lines:
        return None

    raw_items = _split_items(lines)

    # Split into kicks/punches for this rank
    kicks, punches = [], []
    for it in raw_items:
        it_l = _lc(it)
        if "geri" in it_l:
            kicks.append(it)
        elif any(w in it_l for w in ["tsuki", "shuto", "ken", "strike"]):
            punches.append(it)
        else:
            punches.append(it)

    kicks = _dedup(kicks)
    punches = _dedup(punches)

    # ---- CUMULATIVE OPTION: add 9th-kyu foundational kicks only if user asks cumulative
    carry_kicks = []
    if cumulative and rank_key != "9th kyu":
        nine_block = _extract_rank_block(rank_text, "9th kyu")
        if nine_block:
            nine_lines = _extract_section_lines(nine_block, "Striking:")
            nine_items = _split_items(nine_lines)
            for it in nine_items:
                if "geri" in _lc(it):
                    carry_kicks.append(it)
        carry_kicks = _dedup([k for k in carry_kicks if _lc(k) not in {_lc(x) for x in kicks}])

    # Pretty labels with aliases
    kicks_pretty = [_with_kick_aliases(k) for k in kicks]
    punches_pretty = [_with_punch_aliases(p) for p in punches]
    carry_pretty = [_with_kick_aliases(k) for k in carry_kicks]

    # Normalize capitalization for header (avoid “8Th”)
    def _title_rank(s: str) -> str:
        m = re.match(r"(\d+)(st|nd|rd|th)\s+kyu", s, flags=re.I)
        if m:
            num, suf = m.group(1), m.group(2).lower()
            return f"{num}{suf} Kyu"
        return "Shodan" if s.lower() == "shodan" else s.title()

    parts = []
    if wants_kicks and kicks_pretty:
        parts.append(f"{_title_rank(rank_key)} kicks: {_join_human(kicks_pretty)}.")
        if cumulative and carry_pretty:
            parts.append(f"Carryover (foundational): {_join_human(carry_pretty)}.")
    if wants_punches and punches_pretty:
        parts.append(f"{_title_rank(rank_key)} strikes: {_join_human(punches_pretty)}.")

    return " ".join(parts) if parts else None



def try_answer_rank_nage(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    ql = _lc(question)
    if not any(w in ql for w in ["nage", "throw", "throws", "nage waza"]):
        return None

    rank_key = _rank_key_from_question(question)
    if not rank_key:
        return None

    rank_text = _find_rank_text_from_passages(passages)
    if not rank_text:
        return None

    block = _extract_rank_block(rank_text, rank_key)
    if not block:
        return None

    lines = _extract_section_lines(block, "Nage waza:")
    if not lines:
        return None

    items = _dedup(_split_items(lines))
    if not items:
        return None

    return f"{rank_key.title()} throws: {_join_human(items)}."


def try_answer_rank_jime(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    ql = _lc(question)
    if not any(w in ql for w in ["jime", "choke", "chokes", "strangle"]):
        return None

    rank_key = _rank_key_from_question(question)
    if not rank_key:
        return None

    rank_text = _find_rank_text_from_passages(passages)
    if not rank_text:
        return None

    block = _extract_rank_block(rank_text, rank_key)
    if not block:
        return None

    lines = _extract_section_lines(block, "Jime waza:")
    if not lines:
        return None

    items = _dedup(_split_items(lines))
    if not items:
        return None

    return f"{rank_key.title()} chokes: {_join_human(items)}."


def try_answer_rank_requirements(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Summarize the single-rank block when the user asks for 'requirements for X kyu'.
    Keeps output scoped to ONE rank (prevents 'all ranks' dump).
    """
    ql = _lc(question)
    if not any(w in ql for w in ["requirement", "requirements", "what do i need for", "rank checklist"]):
        return None

    rank_key = _rank_key_from_question(question)
    if not rank_key:
        return None

    rank_text = _find_rank_text_from_passages(passages)
    if not rank_text:
        return None

    block = _extract_rank_block(rank_text, rank_key)
    if not block:
        return None

    sections = []

    def add_section(label: str):
        lines = _extract_section_lines(block, label)
        if lines:
            # If inline + list, split items; otherwise join lines
            if any(sep in " ".join(lines) for sep in [",", ";"]):
                content = _join_human(_dedup(_split_items(lines)))
            else:
                content = " ".join(lines)
            content = _norm(content)
            if content:
                sections.append(f"{label} {content}")

    header_line = re.split(r"\r?\n", block, maxsplit=1)[0].strip()
    add_section("Kamae:")
    add_section("Ukemi:")
    add_section("Kaiten:")
    add_section("Taihenjutsu:")
    add_section("Blocking:")
    add_section("Striking:")
    add_section("Kihon Happo:")
    add_section("San Shin no Kata:")
    add_section("Nage waza:")
    add_section("Jime waza:")
    add_section("Kyusho:")
    add_section("Other:")

    if not sections:
        return header_line

    return f"{header_line}\n" + "\n".join(sections)
