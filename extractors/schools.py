# extractors/schools.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import re

# Canonical names + common aliases (expand as needed)
SCHOOL_ALIASES: Dict[str, List[str]] = {
    "Togakure Ryu": [
        "togakure ryu", "togakure-ryu", "togakure ryū", "togakure-ryū",
        "togakure ryu ninpo", "togakure ryu ninpo taijutsu", "togakure"
    ],
    "Gyokko Ryu": [
        "gyokko ryu", "gyokko-ryu", "gyokko ryū", "gyokko-ryū", "gyokko"
    ],
    "Koto Ryu": [
        "koto ryu", "koto-ryu", "koto ryū", "koto-ryū", "koto"
    ],
    "Shinden Fudo Ryu": [
        "shinden fudo ryu", "shinden fudo-ryu", "shinden fudō ryū", "shinden fudō-ryū",
        "shinden fudo", "shinden fudo ryu dakentaijutsu", "shinden fudo ryu jutaijutsu"
    ],
    "Kukishinden Ryu": [
        "kukishinden ryu", "kukishinden-ryu", "kukishinden ryū", "kukishinden-ryū", "kukishinden"
    ],
    "Takagi Yoshin Ryu": [
        "takagi yoshin ryu", "takagi yoshin-ryu", "takagi yōshin ryū", "takagi yōshin-ryū",
        "takagi yoshin", "hoko ryu takagi yoshin ryu", "takagi"
    ],
    "Gikan Ryu": [
        "gikan ryu", "gikan-ryu", "gikan ryū", "gikan-ryū", "gikan"
    ],
    "Gyokushin Ryu": [
        "gyokushin ryu", "gyokushin-ryu", "gyokushin ryū", "gyokushin-ryū", "gyokushin"
    ],
    "Kumogakure Ryu": [
        "kumogakure ryu", "kumogakure-ryu", "kumogakure ryū", "kumogakure-ryū", "kumogakure"
    ],
}

# ---------- Normalization helpers ----------

_MACRON_MAP = str.maketrans({
    "ō": "o", "Ō": "O",
    "ū": "u", "Ū": "U",
    "ā": "a", "Ā": "A",
    "ī": "i", "Ī": "I",
    "ē": "e", "Ē": "E",
    "’": "'", "“": '"', "”": '"',
})

def _norm(s: str) -> str:
    # strip macrons, unify hyphens/spaces, lowercase
    s = (s or "").translate(_MACRON_MAP)
    s = s.replace("\u2010", "-").replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def _looks_like_school_header(line: str) -> bool:
    # Accept "School: X", "School - X", "School – X", and "Togakure Ryu:" patterns
    t = _norm(line)
    return t.startswith("school:") or t.startswith("school -") or t.startswith("school –") or t.endswith(" ryu:") or t.endswith(" ryu :")

def _canon_for_query(question: str) -> Optional[str]:
    qn = _norm(question)
    for canon, aliases in SCHOOL_ALIASES.items():
        tokens = [_norm(canon)] + [_norm(a) for a in aliases]
        if any(tok in qn for tok in tokens):
            return canon
    # fallback: generic "... ryu" mention → let profile try best-effort
    if " ryu" in qn or " ryu?" in qn or " ryu." in qn:
        # last token before "ryu"
        m = re.search(r"([a-z0-9\- ]+)\s+ryu\b", qn)
        if m:
            guess = m.group(1).strip().replace("-", " ")
            # try to match guess to a canon loosely
            for canon in SCHOOL_ALIASES.keys():
                if _norm(canon).startswith(guess):
                    return canon
    return None

# ---------- Slicing & field extraction ----------

_FIELD_KEYS = ["translation", "type", "focus", "weapons", "notes"]

def _slice_school_blocks(blob: str) -> List[Tuple[str, List[str]]]:
    """
    Return list of (header_line, block_lines). A block ends at next header or '---'.
    Assumes the file uses sections like:
        School: <Name>
        Translation: ...
        Type: ...
        Focus: ...
        Weapons: ...
        Notes: ...
        ---
    """
    lines = blob.splitlines()
    idxs = [i for i, ln in enumerate(lines) if _looks_like_school_header(ln)]
    blocks: List[Tuple[str, List[str]]] = []
    for j, i in enumerate(idxs):
        start = i
        end = idxs[j + 1] if j + 1 < len(idxs) else len(lines)
        # stop at first '---' separator if present
        for k in range(i + 1, end):
            if lines[k].strip() == "---":
                end = k
                break
        blocks.append((lines[start], lines[start + 1:end]))
    return blocks

def _header_matches(header_line: str, canon: str) -> bool:
    # Accept "School: Togakure Ryu", "School: Togakure Ryu Ninpo Taijutsu", "Togakure Ryu:"
    h = _norm(header_line)
    cn = _norm(canon)
    # must contain the canonical token (e.g., "togakure ryu")
    return cn in h

def _parse_fields(block_lines: List[str]) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for ln in block_lines:
        if not ln.strip():
            continue
        m = re.match(r"^\s*([A-Za-z][A-Za-z ]{1,20}):\s*(.*)$", ln)
        if m:
            key = _norm(m.group(1))
            val = m.group(2).strip()
            # collapse any indented continuation lines
            data[key] = data.get(key, "") + ((" " if key in data and data[key] else "") + val)
        else:
            # treat as continuation of previous key if any
            if data:
                last_key = list(data.keys())[-1]
                data[last_key] = (data[last_key] + " " + ln.strip()).strip()
    # keep only known keys
    return {k: v.strip() for k, v in data.items() if k in _FIELD_KEYS and v.strip()}

def _format_profile(canon: str, fields: Dict[str, str], bullets: bool) -> str:
    title = canon
    if bullets:
        parts = [f"{title}:"]
        for k in ["translation", "type", "focus", "weapons", "notes"]:
            if k in fields:
                label = k.capitalize()
                parts.append(f"- {label}: {fields[k]}")
        return "\n".join(parts)
    else:
        # concise paragraph
        segs = []
        if "translation" in fields:
            segs.append(f'Translation “{fields["translation"]}”.')
        if "type" in fields:
            segs.append(f"Type: {fields['type']}.")
        if "focus" in fields:
            segs.append(f"Focus: {fields['focus']}.")
        if "weapons" in fields:
            segs.append(f"Weapons: {fields['weapons']}.")
        if "notes" in fields:
            segs.append(f"Notes: {fields['notes']}.")
        if not segs:
            return f"{title}."
        return f"{title}: " + " ".join(segs)

def _find_school_block(blob: str, canon: str) -> Optional[Dict[str, str]]:
    blocks = _slice_school_blocks(blob)
    # try strict header match first
    for header, body in blocks:
        if _header_matches(header, canon):
            fields = _parse_fields(body)
            if fields:
                return fields
    # fuzzy fallback: search within body lines for the canonical token
    cn = _norm(canon)
    for header, body in blocks:
        all_text = _norm(" ".join([header] + body))
        if cn in all_text:
            fields = _parse_fields(body)
            if fields:
                return fields
    return None

# ---------- Public API ----------

def try_answer_school_profile(
    question: str,
    passages: List[Dict[str, Any]],
    *,
    bullets: bool = True,
) -> Optional[str]:
    """
    Deterministic school profile answer using ONLY context from Schools summaries.
    """
    canon = _canon_for_query(question)
    if not canon:
        return None

    # Find a chunk that looks like the Schools summaries (prefer injected synthetic first)
    schools_text = None
    for p in passages:
        src = (p.get("source") or "").lower()
        txt = p.get("text") or ""
        if "schools of the bujinkan summaries" in src:
            schools_text = txt
            break
    if not schools_text:
        # allow cases where the entire file is injected as first hit
        for p in passages:
            txt = p.get("text") or ""
            if "school:" in _norm(txt):
                schools_text = txt
                break
    if not schools_text:
        return None

    fields = _find_school_block(schools_text, canon)
    if not fields:
        # deterministic: return a clear, minimal line so we can see the path
        return f"I am sorry, but there is no information available in the provided text about {canon}."

    return _format_profile(canon, fields, bullets=bullets)
