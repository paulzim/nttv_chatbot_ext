# extractors/leadership.py
import re
from typing import List, Dict, Any, Optional

# Canonical school keys and resilient alias sets (typos included)
SCHOOL_ALIASES = {
    "gyokko-ryu": [
        "gyokko-ryu", "gyokko ryu", "gyokko-ryū", "gyokko ryū",
        "gyokku ryu", "gyokku-ryu", "gyokku ryū"  # common typo
    ],
    "koto-ryu": [
        "koto-ryu", "koto ryu", "koto-ryū", "koto ryū",
        "koto ryu koppojutsu", "koto-ryu koppojutsu"
    ],
    "togakure-ryu": ["togakure-ryu", "togakure ryu", "togakure-ryū", "togakure ryū"],
    "shinden fudo-ryu": ["shinden fudo-ryu", "shinden fudo ryu", "shinden fudō-ryū", "shinden fudō ryū"],
    "kukishinden-ryu": ["kukishinden-ryu", "kukishinden ryu", "kukishinden-ryū", "kukishinden ryū"],
    "takagi yoshin-ryu": ["takagi yoshin-ryu", "takagi yoshin ryu", "takagi yōshin-ryū", "takagi yōshin ryū"],
    "gikan-ryu": ["gikan-ryu", "gikan ryu", "gikan-ryū", "gikan ryū"],
    "gyokushin-ryu": ["gyokushin-ryu", "gyokushin ryu", "gyokushin-ryū", "gyokushin ryū"],
    "kumogakure-ryu": ["kumogakure-ryu", "kumogakure ryu", "kumogakure-ryū", "kumogakure ryū"],
}

QUALIFIERS = [
    # common style descriptors we should ignore when mapping to canonical school
    "koshijutsu", "kosshijutsu",
    "koppojutsu",
    "dakentaijutsu",
    "jutaijutsu",
    "happo bikenjutsu", "happo hikenjutsu", "happō bikenjutsu", "hikenjutsu",
    "ninpo taijutsu", "ninjutsu", "budo taijutsu", "budō taijutsu",
]

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _strip_macrons(s: str) -> str:
    return (s.replace("ō", "o")
             .replace("ū", "u")
             .replace("ā", "a")
             .replace("ī", "i")
             .replace("Ō", "O")
             .replace("Ū", "U")
             .replace("Ā", "A")
             .replace("Ī", "I"))

def _just_school_ryu(s: str) -> str:
    """
    Reduce things like 'Gyokko Ryu Koshijutsu' -> 'Gyokko Ryu'
    and normalize macrons/spacing/case.
    """
    s0 = _strip_macrons(_norm_ws(s)).lower()
    # remove qualifiers if present
    for q in QUALIFIERS:
        s0 = s0.replace(q, "")
    s0 = _norm_ws(s0)
    # take up to "... ryu"
    m = re.search(r"\b([a-z' .]+?\sryu)\b", s0)
    if m:
        return m.group(1)
    # fallback: if it already contains 'ryu' keep left part
    if "ryu" in s0:
        return s0.split("ryu", 1)[0].strip() + " ryu"
    return s0

def _alias_to_key(name_like: str) -> Optional[str]:
    core = _just_school_ryu(name_like)
    for key, aliases in SCHOOL_ALIASES.items():
        for a in aliases:
            if _strip_macrons(a).lower() in core:
                return key
    # loose guess: '<word> ryu'
    m = re.search(r"\b([a-z]+)\s+ryu\b", core)
    if m:
        guess = m.group(0)
        for key, aliases in SCHOOL_ALIASES.items():
            if any(guess in _strip_macrons(x).lower() for x in aliases):
                return key
    return None

def _pretty_school(key: str) -> str:
    return key.replace("-", " ").title().replace("Ryu", "Ryū")

# Accept :, -, – , —  OR pipe-delimited rows (first col=school, second col=person)
SOKESHIP_KV = re.compile(r"^\s*([A-Za-z0-9 .’'ʻ`\-ōūāī]+?)\s*[:\-–—]\s*(.+?)\s*$")

NAT_FORMS = [
    re.compile(r"^\s*(.+?)\s+(?:has\s+been|was|became)\s+(?:named|appointed|designated\s+as\s+)?(?:the\s+)?s[oō]ke\s+of\s+(.+?)\s*\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(.+?)\s+is\s+(?:the\s+)?s[oō]ke\s+of\s+(.+?)\s*\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*s[oō]ke\s+of\s+(.+?)\s+is\s+(.+?)\s*\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(.+?)\s+s[oō]ke\s*[:\-–—]\s*(.+?)\s*$", re.IGNORECASE),
]

def _harvest_pairs_from_text(text: str) -> List[tuple[str, str]]:
    """Return (school_like, person) pairs from leadership-like lines."""
    pairs: List[tuple[str, str]] = []
    lines = (text or "").splitlines()

    # 0) Pipe-delimited table rows (works even without an explicit [SOKESHIP] header)
    for ln in lines:
        if "|" in ln:
            cols = [c.strip() for c in ln.split("|")]
            if len(cols) >= 2 and len(cols[0]) >= 4 and len(cols[1]) >= 2:
                # cols[0] ~ school, cols[1] ~ person
                school_like = _norm_ws(cols[0])
                person = _norm_ws(cols[1])
                # skip obvious non-rows (e.g., separators or accidental pipes)
                if not re.search(r"[A-Za-z]", school_like) or not re.search(r"[A-Za-z]", person):
                    continue
                pairs.append((school_like, person))

    # 1) Key-value lines (":" or dash based)
    for ln in lines:
        m = SOKESHIP_KV.match(ln)
        if m:
            school_like = _norm_ws(m.group(1))
            person = _norm_ws(m.group(2))
            if len(school_like) >= 4 and len(person) >= 2:
                pairs.append((school_like, person))

    # 2) Natural-language forms
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
                school_like = _norm_ws(m.group(1)); person = _norm_ws(m.group(2))
            elif "s[oō]ke\\s*[:" in pat.pattern:
                # "<SCHOOL> Soke: <NAME>"
                school_like = _norm_ws(m.group(1)); person = _norm_ws(m.group(2))
            else:
                # "<NAME> ... Soke of <SCHOOL>"
                person = _norm_ws(m.group(1)); school_like = _norm_ws(m.group(2))
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
      return '<Person> is the current sōke of <School>.' using leadership data.
    """
    ql = _strip_macrons(question.lower())

    if not any(t in ql for t in ["soke", "soke'", "sōke", "grandmaster", "headmaster", "current head", "current grandmaster"]):
        return None

    # tolerate common typos
    ql = ql.replace("gyokku ryu", "gyokko ryu").replace("gyokku-ryu", "gyokko-ryu")

    # Which school is being asked?
    target = None
    for key, aliases in SCHOOL_ALIASES.items():
        if any(_strip_macrons(a).lower() in ql for a in aliases):
            target = key
            break
    if not target:
        # last-ditch: extract "<word> ryu" and try mapping
        m = re.search(r"\b([a-z]+)\s+ryu\b", ql)
        if m:
            target = _alias_to_key(m.group(0))
    if not target:
        return None

    # 1) Aggregate whole leadership text first
    agg = _aggregate_leadership_text(passages)
    pairs = _harvest_pairs_from_text(agg)
    for school_like, person in pairs:
        key = _alias_to_key(school_like)
        if key == target:
            return f"{person} is the current sōke of {_pretty_school(target)}."

    # 2) Fallback: scan all retrieved passages (in case leadership rows appear elsewhere)
    for p in passages:
        txt = p.get("text") or ""
        for school_like, person in _harvest_pairs_from_text(txt):
            key = _alias_to_key(school_like)
            if key == target:
                return f"{person} is the current sōke of {_pretty_school(target)}."

    return None
