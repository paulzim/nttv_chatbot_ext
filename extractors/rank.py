# extractors/rank.py
import re
from typing import List, Dict, Any
from .common import dedupe_preserve

# ---------- shared helpers ----------
# Accepts: "8th kyu", "8 kyu", and "kyu 8"
_ORD_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+kyu\b", re.I)
_REV_RE = re.compile(r"\bkyu\s*(\d{1,2})(?:st|nd|rd|th)?\b", re.I)

def _parse_rank_label(q: str) -> str | None:
    ql = q.lower()
    m = _ORD_RE.search(ql) or _REV_RE.search(ql)
    if not m:
        return None
    num = m.group(1)
    # canonicalize ordinal suffix
    suf = "th"
    if not num.endswith(("11", "12", "13")):
        if num.endswith("1"):
            suf = "st"
        elif num.endswith("2"):
            suf = "nd"
        elif num.endswith("3"):
            suf = "rd"
    return f"{num}{suf} Kyu"

def _split_lines(s: str) -> List[str]:
    return s.replace("\r\n", "\n").replace("\r", "\n").split("\n")

def _is_rank_header(line: str) -> bool:
    l = line.strip().lower()
    return bool(
        _ORD_RE.search(l) or
        re.search(r"\bkyu\s*:\s*\d{1,2}(?:st|nd|rd|th)?\s+kyu\b", l)
    )

def _find_rank_block(full_text: str, rank_label: str) -> str | None:
    """
    Line-based: find the first line that mentions the rank, then collect lines
    until the next rank header or EOF.
    """
    lines = _split_lines(full_text)
    start_idx = None
    pat = re.compile(rf"(?i)\b{re.escape(rank_label)}\b|kyu\s*:\s*{re.escape(rank_label)}\b")
    for i, ln in enumerate(lines):
        if pat.search(ln):
            start_idx = i
            break
    if start_idx is None:
        return None

    out = [lines[start_idx]]
    for ln in lines[start_idx + 1:]:
        if _is_rank_header(ln):
            break
        out.append(ln)
    return "\n".join(out).strip()

def _split_items(s: str) -> List[str]:
    """Split tolerant to commas/semicolons/slashes/newlines/bullets/pipes/“and”."""
    if not s:
        return []
    s = re.sub(r"\band\b", ",", s, flags=re.I)
    s = s.replace("•", ",").replace("・", ",").replace("|", ",")
    parts = re.split(r"[,\n;\/]+", s)
    out = []
    for p in parts:
        t = p.strip().strip("-–—").strip()
        if not t:
            continue
        t = re.sub(r"^\((.*)\)$", r"\1", t).strip()
        t = re.sub(r"\s+", " ", t)
        out.append(t)
    return dedupe_preserve(out)

def _pick_rank_passages(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rp = [p for p in passages if "rank requirements" in (p.get("source") or "").lower()]
    return rp if rp else passages

# ---------- STRIKING (kicks/punches) ----------
_STRIKE_HEAD_RE = re.compile(r"(?i)^\s*striking(?:\s+\w+){0,3}\s*(?::|[-–—])?\s*(.*)$")

def _extract_striking_line(block_text: str) -> str | None:
    for ln in _split_lines(block_text):
        m = _STRIKE_HEAD_RE.match(ln)
        if m:
            rest = m.group(1).strip()
            if rest:
                return rest
    return None

def _is_kick(x: str) -> bool:
    lx = x.lower().strip()
    WHITELIST = {
        "zenpo geri", "sokuho geri", "koho geri", "sakui geri", "happo geri"
    }
    return ("geri" in lx) or (" kick" in lx) or lx.endswith("kick") or lx in WHITELIST

def _is_punch(x: str) -> bool:
    lx = x.lower()
    return ("tsuki" in lx) or lx.endswith(" ken") or (" ken " in lx) or ("uraken" in lx)

_KICK_INTENT   = re.compile(r"\b(kick|kicks|geri|geris)\b", re.I)
_PUNCH_INTENT  = re.compile(r"\b(punch|punches|tsuki|tsukis)\b|\b(?:^|[\s\-])ken\b", re.I)
_STRIKE_INTENT = re.compile(r"\b(strike|strikes|striking)\b", re.I)

def try_answer_rank_striking(question: str, passages: List[Dict[str, Any]]) -> str | None:
    ql = question.lower()
    rank_label = _parse_rank_label(ql)
    if not rank_label:
        return None

    wants_kicks   = bool(_KICK_INTENT.search(ql))
    wants_punches = bool(_PUNCH_INTENT.search(ql))
    wants_strikes = bool(_STRIKE_INTENT.search(ql)) or (not wants_kicks and not wants_punches)

    rank_passages = _pick_rank_passages(passages)
    full_text = "\n\n".join(p["text"] for p in rank_passages)

    block = _find_rank_block(full_text, rank_label)
    if not block:
        return None

    striking = _extract_striking_line(block)
    tokens = _split_items(striking) if striking else _split_items(block)

    kicks   = [x for x in tokens if _is_kick(x)]
    punches = [x for x in tokens if _is_punch(x)]

    parts: List[str] = []
    if wants_kicks or (wants_strikes and kicks):
        parts.append(f"{rank_label} kicks: " + (", ".join(kicks) if kicks else "(none listed)") + ".")
    if wants_punches or (wants_strikes and punches):
        parts.append(f"{rank_label} punches: " + (", ".join(punches) if punches else "(none listed)") + ".")

    if parts:
        return " ".join(parts)
    if wants_kicks and not kicks:
        return f"{rank_label} kicks: (none listed)."
    if wants_punches and not punches:
        return f"{rank_label} punches: (none listed)."
    return None

# ---------- NAGE WAZA (throws) ----------
# Accept "Nage waza:", "Nage-waza:", "Throws:", etc.
_NAGE_HEAD_RE = re.compile(r"(?i)^\s*(?:nage\s*-?\s*waza|throws)\s*(?::|[-–—])?\s*(.*)$")
# Query intents that imply throws (synonyms included)
_THROW_INTENT = re.compile(r"\b(throw|throws|toss|nage|projection|take\s*down|takedown)\b", re.I)

def _extract_nage_line(block_text: str) -> str | None:
    for ln in _split_lines(block_text):
        m = _NAGE_HEAD_RE.match(ln)
        if m:
            rest = m.group(1).strip()
            if rest:
                return rest
    return None

def try_answer_rank_nage(question: str, passages: List[Dict[str, Any]]) -> str | None:
    ql = question.lower()
    if not _THROW_INTENT.search(ql) and "nage" not in ql:
        return None

    rank_label = _parse_rank_label(ql)
    if not rank_label:
        return None

    rank_passages = _pick_rank_passages(passages)
    full_text = "\n\n".join(p["text"] for p in rank_passages)
    block = _find_rank_block(full_text, rank_label)
    if not block:
        return None

    # Prefer a dedicated "Nage waza"/"Throws" line; if absent, scan whole block
    nage = _extract_nage_line(block)
    items = _split_items(nage) if nage else _split_items(block)

    # Heuristic: if a "Nage waza" line existed but had no items, say (none listed)
    if nage is not None and not items:
        return f"{rank_label} throws (Nage waza): (none listed)."

    if items:
        return f"{rank_label} throws (Nage waza): " + ", ".join(items) + "."

    return None

# ---------- JIME WAZA (chokes) ----------
_JIME_HEAD_RE = re.compile(r"(?i)^\s*(?:jime\s*-?\s*waza|chokes?)\s*(?::|[-–—])?\s*(.*)$")
# Query intents that imply chokes
_CHOKE_INTENT = re.compile(r"\b(choke|chokes|strangle|jime|strangulation)\b", re.I)

def _extract_jime_line(block_text: str) -> str | None:
    for ln in _split_lines(block_text):
        m = _JIME_HEAD_RE.match(ln)
        if m:
            rest = m.group(1).strip()
            if rest:
                return rest
    return None

def try_answer_rank_jime(question: str, passages: List[Dict[str, Any]]) -> str | None:
    ql = question.lower()
    if not _CHOKE_INTENT.search(ql) and "jime" not in ql:
        return None

    rank_label = _parse_rank_label(ql)
    if not rank_label:
        return None

    rank_passages = _pick_rank_passages(passages)
    full_text = "\n\n".join(p["text"] for p in rank_passages)
    block = _find_rank_block(full_text, rank_label)
    if not block:
        return None

    # Prefer a dedicated "Jime waza/Chokes" line; if absent, scan whole block
    jime = _extract_jime_line(block)
    items = _split_items(jime) if jime else _split_items(block)

    if jime is not None and not items:
        return f"{rank_label} chokes (Jime waza): (none listed)."

    if items:
        return f"{rank_label} chokes (Jime waza): " + ", ".join(items) + "."

    return None

# ---------- WEAPONS (Buki) ----------
# Accept "Weapons:", "Weapon:", "Buki:", etc.
_WEAPON_HEAD_RE = re.compile(r"(?i)^\s*(?:weapons?|buki)\s*(?::|[-–—])?\s*(.*)$")
# Trigger on generic "weapons/buki" or common weapon names to be user-friendly
_WEAPON_INTENT = re.compile(
    r"\b(weapon|weapons|buki|sword|katana|tanto|bokken|bo|hanbo|staff|stick|yari|naginata|shuriken|kusari(?:fundo)?|rope|shoge|kyoketsu(?:\s*shoge)?)\b",
    re.I
)

def _extract_weapon_line(block_text: str) -> str | None:
    for ln in _split_lines(block_text):
        m = _WEAPON_HEAD_RE.match(ln)
        if m:
            rest = m.group(1).strip()
            if rest:
                return rest
    return None

def try_answer_rank_weapons(question: str, passages: List[Dict[str, Any]]) -> str | None:
    ql = question.lower()
    if not _WEAPON_INTENT.search(ql):
        return None

    rank_label = _parse_rank_label(ql)
    if not rank_label:
        return None

    rank_passages = _pick_rank_passages(passages)
    full_text = "\n\n".join(p["text"] for p in rank_passages)
    block = _find_rank_block(full_text, rank_label)
    if not block:
        return None

    # Prefer a dedicated "Weapons/Buki" line; if absent, scan whole block
    wl = _extract_weapon_line(block)
    items = _split_items(wl) if wl else _split_items(block)

    if wl is not None and not items:
        return f"{rank_label} weapons: (none listed)."

    if items:
        return f"{rank_label} weapons: " + ", ".join(items) + "."

    return None
