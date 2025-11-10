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

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _alias_to_key(name_like: str) -> Optional[str]:
    s = name_like.lower().strip()
    # normalize macrons & extras
    repl = (
        ("koppojutsu",""), ("koppōjutsu",""), ("ryū","ryu"),
        ("ō","o"), ("ū","u"), ("ā","a"), ("ī","i"), ("é","e")
    )
    for a, b in repl:
        s = s.replace(a, b)
    # exact alias hit
    for key, aliases in SCHOOL_ALIASES.items():
        for a in aliases:
            if a.replace("ryū","ryu") in s:
                return key
    # loose: "<word> ryu"
    m = re.search(r"\b([a-z]+)\s+ryu\b", s)
    if m:
        guess = m.group(0)
        for key, aliases in SCHOOL_ALIASES.items():
            if any(guess in x.replace("ryū","ryu") for x in aliases):
                return key
    return None

def _pretty_school(key: str) -> str:
    return key.replace("-", " ").title().replace("Ryu", "Ryū")

# Tolerant KV separators for SOKESHIP-like lines (allow :, -, – , —)
SOKESHIP_KV = re.compile(r"^\s*([A-Za-z0-9 .’'ʻ`\-ōūāīÉé]+?)\s*[:\-–—]\s*(.+?)\s*$")

# Natural statements (many flexible forms)
NAT_FORMS = [
    # "<NAME> has been/was/became (the) Soke/Sōke of <SCHOOL>"
    re.compile(r"^\s*(.+?)\s+(?:has\s+been|was|became)\s+(?:named|appointed|designated\s+as\s+)?(?:the\s+)?s[oō]ke\s+of\s+(.+?)\s*\.?\s*$", re.IGNORECASE),
    # "<NAME> is (the) Soke/Sōke of <SCHOOL>"
    re.compile(r"^\s*(.+?)\s+is\s+(?:the\s+)?s[oō]ke\s+of\s+(.+?)\s*\.?\s*$", re.IGNORECASE),
    # "Soke/Sōke of <SCHOOL> is <NAME>"
    re.compile(r"^\s*s[oō]ke\s+of\s+(.+?)\s+is\s+(.+?)\s*\.?\s*$", re.IGNORECASE),
    # "<SCHOOL> Soke/Sōke: <NAME>"
    re.compile(r"^\s*(.+?)\s+s[oō]ke\s*[:\-–—]\s*(.+?)\s*$", re.IGNORECASE),
]

def _harvest_pairs_from_text(text: str) -> List[tuple[str, str]]:
    """Return (school_like, person) pairs from any leadership-like lines in the given text."""
    pairs: List[tuple[str, str]] = []
    lines = (text or "").splitlines()

    # 1) Key-value lines (works with or without a [SOKESHIP] header)
    for ln in lines:
        m = SOKESHIP_KV.match(ln)
        if m:
            school_like = _norm(m.group(1))
            person = _norm(m.group(2))
            # Heuristics: avoid false positives like "Notes - something"
            if len(school_like) >= 4 and len(person) >= 2:
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
                school_like = _norm(m.group(1)); person = _norm(m.group(2))
            elif "s[oō]ke\\s*[:" in pat.pattern:
                # "<SCHOOL> Soke: <NAME>"
                school_like = _norm(m.group(1)); person = _norm(m.group(2))
            else:
                # "<NAME> ... Soke of <SCHOOL>"
                person = _norm(m.group(1)); school_like = _norm(m.group(2))
            if len(school_like) >= 4 and len(person) >= 2:
                pairs.append((school_like, person))

    return pairs

def _aggregate_leadership_text(passages: List[Dict[str, Any]]) -> str:
    """Concatenate ALL chunks from the leadership file so chunk boundaries don't hide lines."""
    blobs = []
    for p in passages:
        src = (p.get("source") or "").lower()
        if "leadership" in src:
            t = p.get("text") or ""
            if t:
                blobs.append(t)
    return "\n".join(blobs)

def try_extract_answer(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic leadership answer:
      If the question mentions soke/grandmaster and a school (even with typos),
      return 'X is the current sōke of <School>.'
    """
    ql = question.lower()
    if not any(t in ql for t in ["soke", "sōke", "grandmaster", "headmaster", "current head", "current grandmaster"]):
        return None

    # Light typo normalization before alias matching
    ql = ql.replace("gyokku ryu", "gyokko ryu").replace("gyokku-ryu", "gyokko-ryu")

    target = None
    for key, aliases in SCHOOL_ALIASES.items():
        if any(a in ql for a in aliases):
            target = key
            break
    if not target:
        return None

    # 1) Aggregate leadership text across ALL its chunks, then parse
    agg = _aggregate_leadership_text(passages)
    pairs = _harvest_pairs_from_text(agg)
    for school_like, person in pairs:
        key = _alias_to_key(school_like)
        if key == target:
            return f"{person} is the current sōke of {_pretty_school(target)}."

    # 2) Fallback: scan ALL passages (in case lines live outside the leadership file)
    for p in passages:
        txt = p.get("text") or ""
        for school_like, person in _harvest_pairs_from_text(txt):
            key = _alias_to_key(school_like)
            if key == target:
                return f"{person} is the current sōke of {_pretty_school(target)}."

    return None
