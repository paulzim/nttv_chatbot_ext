# extractors/leadership.py
import re
from typing import List, Dict, Any, Optional

ALIASES = {
    "koto ryu": ["koto ryu", "koto-ryu", "koto ryū", "koto-ryū", "koto ryu koppojutsu", "koto-ryu koppojutsu"],
    "gyokko ryu": ["gyokko ryu", "gyokko-ryu", "gyokko ryū", "gyokko-ryū", "gyokko ryu koshijutsu"],
    "shinden fudo ryu": ["shinden fudo ryu", "shinden fudō ryū", "shinden fudo-ryu", "shinden fudo ryu dakentaijutsu"],
    "kukishinden ryu": ["kukishinden ryu", "kukishinden-ryu", "kukishinden ryū"],
    "takagi yoshin ryu": ["takagi yoshin ryu", "takagi yōshin ryū", "takagi yoshin-ryu", "takagi yoshin ryu jutaijutsu"],
    "gikan ryu": ["gikan ryu", "gikan-ryu", "gikan ryū", "gikan ryu koppojutsu"],
    "gyokushin ryu": ["gyokushin ryu", "gyokushin-ryu", "gyokushin ryū"],
    "kumogakure ryu": ["kumogakure ryu", "kumogakure-ryu", "kumogakure ryū"],
    "togakure ryu": ["togakure ryu", "togakure-ryu", "togakure ryū"],
}

SOKE_TRIGGERS = [
    "soke", "grandmaster", "current head", "current headmaster", "current grandmaster",
    "who leads", "who is the head", "who is the sōke", "who is sōke",
]

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def _parse_sokeship_block(text: str) -> List[Dict[str, str]]:
    rows = []
    lines = text.splitlines()
    in_block = False
    for ln in lines:
        l = ln.strip()
        if not in_block:
            if l.lower() == "[sokeship]":
                in_block = True
            continue
        if not l:
            break
        parts = [p.strip() for p in l.split("|")]
        if len(parts) >= 2:
            school = parts[0]
            soke = parts[1]
            since = parts[2] if len(parts) >= 3 else ""
            notes = parts[3] if len(parts) >= 4 else ""
            rows.append({"school": school, "soke": soke, "since": since, "notes": notes})
    return rows

def _which_school(question: str) -> Optional[str]:
    ql = question.lower()
    for canonical, alias_list in ALIASES.items():
        if any(a in ql for a in alias_list):
            return canonical
    return None

def try_extract_answer(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    ql = question.lower()
    if not any(t in ql for t in SOKE_TRIGGERS):
        return None

    target = _which_school(ql)
    if not target:
        # If they asked “current grandmasters of Bujinkan”, bail (let other explainers handle)
        return None

    # Search for a [SOKESHIP] block in high-priority passages first
    texts = []
    for p in passages:
        t = (p.get("text") or "")
        if "[SOKESHIP]" in t:
            texts.append(t)

    # Fallback: any passage from leadership file
    if not texts:
        for p in passages:
            src = (p.get("source") or "").lower()
            if "leadership" in src:
                texts.append(p.get("text") or "")

    if not texts:
        return None

    table = []
    for t in texts:
        table.extend(_parse_sokeship_block(t))
    if not table:
        return None

    # Greedy match: canonical name containment
    for row in table:
        school_norm = row["school"].lower()
        if target.split()[0] in school_norm or target in school_norm:
            soke = row["soke"].strip()
            since = row["since"].strip()
            if soke:
                when = f" (since {since})" if since else ""
                # One-sentence fact; app’s explanation pipeline will elaborate.
                return f"{row['school']}: {soke} Sensei is the current Soke{when}."
    return None
