# extractors/leadership.py
import re
from typing import List, Dict, Any, Optional

# Canonical school keys and resilient alias sets (typos included)
SCHOOL_ALIASES = {
    "gyokko-ryu": [
        "gyokko-ryu", "gyokko ryu", "gyokko-ryū", "gyokko ryū",
        "gyokku ryu", "gyokku-ryu", "gyokku ryū"  # common typo
    ],
    "koto-ryu": ["koto-ryu", "koto ryu", "koto-ryū", "koto ryū", "koto ryu koppojutsu", "koto-ryu koppojutsu"],
    "togakure-ryu": ["togakure-ryu", "togakure ryu", "togakure-ryū", "togakure ryū"],
    "shinden fudo-ryu": ["shinden fudo-ryu", "shinden fudo ryu", "shinden fudō-ryū", "shinden fudō ryū"],
    "kukishinden-ryu": ["kukishinden-ryu", "kukishinden ryu", "kukishinden-ryū", "kukishinden ryū"],
    "takagi yoshin-ryu": ["takagi yoshin-ryu", "takagi yoshin ryu", "takagi yōshin-ryū", "takagi yōshin ryū"],
    "gikan-ryu": ["gikan-ryu", "gikan ryu", "gikan-ryū", "gikan ryū"],
    "gyokushin-ryu": ["gyokushin-ryu", "gyokushin ryu", "gyokushin-ryū", "gyokushin ryū"],
    "kumogakure-ryu": ["kumogakure-ryu", "kumogakure ryu", "kumogakure-ryū", "kumogakure ryū"],
}

# Normalization helpers
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _alias_to_key(name_like: str) -> Optional[str]:
    s = name_like.lower().strip()
    s = s.replace("koppojutsu", "").replace("koppōjutsu", "")
    s = s.replace("ryū", "ryu").replace("ō", "o").replace("ū", "u").replace("ā", "a").replace("ī", "i").replace("é","e")
    for key, aliases in SCHOOL_ALIASES.items():
        for a in aliases:
            a_norm = a.replace("ryū", "ryu")
            if a_norm in s:
                return key
    # loose fallback: match "<word> ryu" tokens
    m = re.search(r"\b([a-z]+)\s+ryu\b", s)
    if m:
        guess = m.group(0)
        for key, aliases in SCHOOL_ALIASES.items():
            if any(guess in x.replace("ryū","ryu") for x in aliases):
                return key
    return None

def _pretty_school(key: str) -> str:
    return key.replace("-", " ").title().replace("Ryu", "Ryū")

# Patterns
SOKESHIP_BLOCK = re.compile(r"^\s*\[SOKESHIP\]\s*$", re.IGNORECASE)
# Allow colon, hyphen, en/em dash as separators
SOKESHIP_KV = re.compile(r"^\s*([A-Za-z0-9 .’'ʻ`\-ōūāīÉé]+?)\s*[:\-–—]\s*(.+?)\s*$")

# Natural statements (multiple flexible forms)
NAT_FORMS = [
    # "<NAME> has been (the) Soke/Sōke of <SCHOOL>"
    re.compile(r"^\s*(.+?)\s+has\s+been\s+(?:named|appointed|designated)?\s*(?:the\s+)?s[oō]ke\s+of\s+(.+?)\s*\.?\s*$", re.IGNORECASE),
    # "<NAME> is (the) Soke/Sōke of <SCHOOL>"
    re.compile(r"^\s*(.+?)\s+is\s+(?:the\s+)?s[oō]ke\s+of\s+(.+?)\s*\.?\s*$", re.IGNORECASE),
    # "Soke/Sōke of <SCHOOL> is <NAME>"
    re.compile(r"^\s*s[oō]ke\s+of\s+(.+?)\s+is\s+(.+?)\s*\.?\s*$", re.IGNORECASE),
    # "<SCHOOL> Soke/Sōke: <NAME>"
    re.compile(r"^\s*(.+?)\s+s[oō]ke\s*[:\-–—]\s*(.+?)\s*$", re.IGNORECASE),
    # "<NAME> (was|became) (the) Soke/Sōke of <SCHOOL>"
    re.compile(r"^\s*(.+?)\s+(?:was|became)\s+(?:the\s+)?s[oō]ke\s+of\s+(.+?)\s*\.?\s*$", re.IGNORECASE),
]

def _find_pairs_in_text(text: str) -> List[tuple[str, str]]:
    """Return list of (school_like, person) pairs discovered in text (both SOKESHIP block and natural lines)."""
    pairs: List[tuple[str, str]] = []
    lines = (text or "").splitlines()

    # 1) [SOKESHIP] table parsing
    in_block = False
    for ln in lines:
        if SOKESHIP_BLOCK.match(ln):
            in_block = True
            continue
        if in_block:
            if not ln.strip():
                # blank line ends block
                in_block = False
                continue
            m = SOKESHIP_KV.match(ln)
            if m:
                school_like = _norm(m.group(1))
                person = _norm(m.group(2))
                if school_like and person:
                    pairs.append((school_like, person))

    # 2) Natural statements
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        for pat in NAT_FORMS:
            m = pat.match(s)
            if not m:
                continue
            if pat.pattern.startswith("^\\s*s"):
                # "Soke of <SCHOOL> is <NAME>"
                school_like = _norm(m.group(1))
                person = _norm(m.group(2))
            elif "s[oō]ke\\s*[:" in pat.pattern:
                # "<SCHOOL> Soke: <NAME>"
                school_like = _norm(m.group(1))
                person = _norm(m.group(2))
            else:
                # "<NAME> ... Soke of <SCHOOL>"
                person = _norm(m.group(1))
                school_like = _norm(m.group(2))
            if school_like and person:
                pairs.append((school_like, person))

    return pairs

def try_extract_answer(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic leadership answer:
      - If question mentions soke/grandmaster AND a school (even with typos),
        return 'X is the current sōke of <School>.'
    Searches leadership file first, then all passages. Tolerant to punctuation and phrasing.
    """
    ql = question.lower()
    if not any(t in ql for t in ["soke", "sōke", "grandmaster", "headmaster", "current head", "current grandmaster"]):
        return None

    target = None
    # Light typo normalization before alias matching
    ql = ql.replace("gyokku ryu", "gyokko ryu").replace("gyokku-ryu", "gyokko-ryu")
    for key, aliases in SCHOOL_ALIASES.items():
        if any(a in ql for a in aliases):
            target = key
            break
    if not target:
        return None

    # Prefer leadership file(s)
    prioritized = sorted(passages, key=lambda p: 0 if "leadership" in (p.get("source","").lower()) else 1)

    # 1) Leadership file pass
    for p in prioritized:
        txt = p.get("text") or ""
        pairs = _find_pairs_in_text(txt)
        for school_like, person in pairs:
            key = _alias_to_key(school_like)
            if key == target:
                return f"{person} is the current sōke of {_pretty_school(target)}."

    # 2) Fallback: scan all passages (in case info is elsewhere)
    for p in passages:
        txt = p.get("text") or ""
        pairs = _find_pairs_in_text(txt)
        for school_like, person in pairs:
            key = _alias_to_key(school_like)
            if key == target:
                return f"{person} is the current sōke of {_pretty_school(target)}."

    return None
