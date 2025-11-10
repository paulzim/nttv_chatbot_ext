# extractors/schools.py
import re
from typing import List, Dict, Any, Optional

# Canonical keys + aliases (ASCII/macrons + common typos)
SCHOOL_ALIASES = {
    "togakure-ryu": [
        "togakure-ryu", "togakure ryu", "togakure-ryū", "togakure ryū",
        "togakure", "togakure ryu ninpo", "togakure-ryu ninpo", "togakure ryu ninpo taijutsu"
    ],
    "gyokko-ryu":   ["gyokko-ryu", "gyokko ryu", "gyokko-ryū", "gyokko ryū", "gyokko", "gyokko ryu kosshijutsu"],
    "koto-ryu":     ["koto-ryu", "koto ryu", "koto-ryū", "koto ryū", "koto", "koto ryu koppojutsu"],
    "shinden fudo-ryu": [
        "shinden fudo-ryu", "shinden fudo ryu", "shinden fudō-ryū", "shinden fudō ryū",
        "shinden fudo", "shinden fudo ryu dakentaijutsu", "shinden fudo ryu jutaijutsu"
    ],
    "kukishinden-ryu":  ["kukishinden-ryu", "kukishinden ryu", "kukishinden-ryū", "kukishinden ryū", "kukishinden"],
    "takagi yoshin-ryu": [
        "takagi yoshin-ryu", "takagi yoshin ryu", "takagi yōshin-ryū", "takagi yōshin ryū", "takagi yoshin"
    ],
    "gikan-ryu":    ["gikan-ryu", "gikan ryu", "gikan-ryū", "gikan ryū", "gikan"],
    "gyokushin-ryu": ["gyokushin-ryu", "gyokushin ryu", "gyokushin-ryū", "gyokushin ryū", "gyokushin", "gyokshin"],
    "kumogakure-ryu": ["kumogakure-ryu", "kumogakure ryu", "kumogakure-ryū", "kumogakure ryū", "kumogakure"],
}

KEY_SENTENCE_HINTS = [
    "ninpo", "ninjutsu", "taijutsu",
    "kosshijutsu", "koppojutsu", "dakentaijutsu", "jutaijutsu",
    "stealth", "infiltration", "concealment",
    "distance", "timing", "kamae", "footwork",
    "weapons", "strategy", "philosophy", "lineage", "soke", "sōke",
]

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _strip_macrons(s: str) -> str:
    return (s or "").replace("ō", "o").replace("ū", "u").replace("ā", "a").replace("ī", "i") \
                    .replace("Ō","O").replace("Ū","U").replace("Ā","A").replace("Ī","I")

def _pretty_school(key: str) -> str:
    return key.replace("-", " ").title().replace("Ryu", "Ryū")

def _target_key_from_question(question: str) -> Optional[str]:
    q = _strip_macrons(question.lower())
    for key, aliases in SCHOOL_ALIASES.items():
        for a in aliases:
            if _strip_macrons(a).lower() in q:
                return key
    m = re.search(r"\b([a-z]+)\s+ryu\b", q)
    if m:
        guess = m.group(0)
        for key, aliases in SCHOOL_ALIASES.items():
            if any(guess in _strip_macrons(x).lower() for x in aliases):
                return key
    return None

def _best_schools_blob(passages: List[Dict[str, Any]]) -> Optional[str]:
    # Prefer the concatenated Schools summaries (synthetic injection)
    for p in passages:
        src = (p.get("source") or "").lower()
        if "schools of the bujinkan summaries" in src:
            return p.get("text") or ""
    # Else pick the most “school-dense” passage
    best, best_score = None, -1
    for p in passages:
        txt = p.get("text") or ""
        low = _strip_macrons(txt.lower())
        score = sum(any(a in low for a in aliases) for aliases in SCHOOL_ALIASES.values())
        if score > best_score:
            best, best_score = txt, score
    return best

def _slice_single_school(blob: str, target_key: str) -> Optional[str]:
    """Find first alias for target; cut to next other-school alias or double newline; cap length."""
    if not blob or target_key not in SCHOOL_ALIASES:
        return None
    text = blob
    low = _strip_macrons(text.lower())

    aliases = [_strip_macrons(a).lower() for a in SCHOOL_ALIASES[target_key]]
    starts = [low.find(a) for a in aliases if a in low]
    if not starts:
        return None
    start = min(i for i in starts if i >= 0)

    # Next other school alias
    other_aliases = []
    for key, als in SCHOOL_ALIASES.items():
        if key == target_key:
            continue
        other_aliases.extend(_strip_macrons(a).lower() for a in als)

    next_alias_pos = len(text)
    for a in other_aliases:
        i = low.find(a, start + 1)
        if i != -1 and i < next_alias_pos:
            next_alias_pos = i

    # Next strong break (double newline)
    m_break = re.search(r"\n\s*\n", text[start:])
    next_break_pos = (start + m_break.start()) if m_break else len(text)

    end = min(next_alias_pos, next_break_pos, start + 1600)
    section = text[start:end].strip()
    return section if section else None

def _sentences(s: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9‘“])", _norm(s))
    return [p for p in parts if len(p) > 5]

def _score_sentence_for_school_info(s: str) -> float:
    low = _strip_macrons(s.lower())
    score = 0.0
    for kw in KEY_SENTENCE_HINTS:
        if kw in low:
            score += 1.0
    # prefer sentences that include “focus/known for/emphasizes”
    if any(x in low for x in ["focus", "known for", "emphasiz", "specializ", "features"]):
        score += 0.5
    # discourage meta/organizational generic lines
    if any(x in low for x in ["organization", "founded in", "international", "based in"]):
        score -= 1.5
    return score

def try_answer_school_profile(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Return 2–4 concrete sentences describing the specific school asked about,
    cut from the Schools summaries block only (no bleed to other schools).
    """
    target = _target_key_from_question(question)
    if not target:
        return None

    blob = _best_schools_blob(passages)
    section = _slice_single_school(blob or "", target)
    if not section:
        return None

    sents = _sentences(section)
    if not sents:
        return None

    # Score sentences for concreteness and pick top 3–4
    scored = sorted((( _score_sentence_for_school_info(s), s) for s in sents), reverse=True)
    top = [s for sc, s in scored[:4] if sc > 0.0] or sents[:3]

    pretty = _pretty_school(target)
    body = " ".join(top[:4])
    if not _strip_macrons(body.lower()).startswith(_strip_macrons(pretty.lower())):
        body = f"{pretty}: {body}"
    return body

def try_answer_schools(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    q = _strip_macrons(question.lower())
    if not (("school" in q) or ("schools" in q) or ("bujinkan" in q)):
        return None

    # Try to find a compact list in context
    for p in passages:
        txt = p.get("text") or ""
        low = _strip_macrons(txt.lower())
        if all(any(a in low for a in aliases) for aliases in SCHOOL_ALIASES.values()):
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            joined = " ".join(lines)
            joined = re.sub(r"\s+", " ", joined)
            if len(joined) > 400:
                joined = joined[:400].rsplit(" ", 1)[0] + "…"
            return joined

    # Fallback canonical list
    return ("The Bujinkan encompasses nine classical ryū: Togakure-ryū, Gyokushin-ryū, Kumogakure-ryū, "
            "Gikan-ryū, Gyokko-ryū, Koto-ryū, Shinden Fudō-ryū, Kukishinden-ryū, and Takagi Yōshin-ryū.")
