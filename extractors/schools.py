import re
from typing import List, Dict, Any, Optional

# Canonical keys + aliases (ASCII/macrons + common typos)
SCHOOL_ALIASES = {
    "togakure-ryu": [
        "togakure-ryu","togakure ryu","togakure-ryū","togakure ryū","togakure",
        "togakure ninpo taijutsu","togakure-ryu ninpo","togakure ryu ninpo taijutsu"
    ],
    "gyokko-ryu":   ["gyokko-ryu","gyokko ryu","gyokko-ryū","gyokko ryū","gyokko","gyokko ryu kosshijutsu"],
    "koto-ryu":     ["koto-ryu","koto ryu","koto-ryū","koto ryū","koto","koto ryu koppojutsu"],
    "shinden fudo-ryu": [
        "shinden fudo-ryu","shinden fudo ryu","shinden fudō-ryū","shinden fudō ryū",
        "shinden fudo","shinden fudo ryu dakentaijutsu","shinden fudo ryu jutaijutsu"
    ],
    "kukishinden-ryu":  ["kukishinden-ryu","kukishinden ryu","kukishinden-ryū","kukishinden ryū","kukishinden"],
    "takagi yoshin-ryu": ["takagi yoshin-ryu","takagi yoshin ryu","takagi yōshin-ryū","takagi yōshin ryū","takagi yoshin"],
    "gikan-ryu":   ["gikan-ryu","gikan ryu","gikan-ryū","gikan ryū","gikan"],
    "gyokushin-ryu":["gyokushin-ryu","gyokushin ryu","gyokushin-ryū","gyokushin ryū","gyokushin","gyokshin"],
    "kumogakure-ryu":["kumogakure-ryu","kumogakure ryu","kumogakure-ryū","kumogakure ryū","kumogakure"],
}

def _strip_macrons(s: str) -> str:
    return (s or "").translate(str.maketrans({"ō":"o","ū":"u","ā":"a","ī":"i","Ō":"O","Ū":"U","Ā":"A","Ī":"I"}))

def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _pretty(key: str) -> str:
    return key.replace("-", " ").title().replace("Ryu", "Ryū")

def _target_from_question(q: str) -> Optional[str]:
    q = _strip_macrons(q.lower())
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

# ---------- Prefer the structured schema if present ----------
SCHEMA_HDR = re.compile(r"(?im)^\s*SCHOOL:\s*(.+)$")
FIELD_RX = {
    "ALIASES": re.compile(r"(?im)^\s*ALIASES:\s*(.+)$"),
    "TRANSLATION": re.compile(r"(?im)^\s*TRANSLATION:\s*(.+)$"),
    "TYPE": re.compile(r"(?im)^\s*TYPE:\s*(.+)$"),
    "FOCUS": re.compile(r"(?im)^\s*FOCUS:\s*(.+)$"),
    "WEAPONS": re.compile(r"(?im)^\s*WEAPONS:\s*(.+)$"),
    "KEY": re.compile(r"(?im)^\s*KEY POINTS:\s*$"),
}

def _schools_blob(passages: List[Dict[str, Any]]) -> Optional[str]:
    # Prefer the concatenated Schools summaries (injected), else “densest” passage
    for p in passages:
        src = (p.get("source") or "").lower()
        if "schools of the bujinkan summaries" in src:
            return p.get("text") or ""
    best, best_score = None, -1
    for p in passages:
        txt = p.get("text") or ""
        low = _strip_macrons(txt.lower())
        score = sum(any(a in low for a in als) for als in SCHOOL_ALIASES.values())
        if score > best_score:
            best, best_score = txt, score
    return best

def _parse_schema_sections(blob: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse all structured SCHOOL blocks into a dict keyed by canonical name (lowercase, ascii).
    """
    if not blob:
        return {}
    sections: Dict[str, Dict[str, Any]] = {}
    # Find every SCHOOL: header and take until next SCHOOL: or EOF
    indices = [m.start() for m in SCHEMA_HDR.finditer(blob)]
    if not indices:
        return sections
    indices.append(len(blob))
    for i in range(len(indices)-1):
        chunk = blob[indices[i]:indices[i+1]]
        # SCHOOL name
        m_name = SCHEMA_HDR.search(chunk)
        if not m_name: 
            continue
        school_name = _strip_macrons(m_name.group(1)).strip()
        key = None
        # map to canonical key
        s_low = school_name.lower()
        for k, als in SCHOOL_ALIASES.items():
            if any(_strip_macrons(a).lower() in s_low for a in als) or _strip_macrons(_pretty(k)).lower() in s_low:
                key = k; break
        if not key:
            # try exact normalized
            key = _strip_macrons(school_name.lower()).replace(" ", "-")
        # fields
        data: Dict[str, Any] = {"RAW": chunk}
        for fname, rx in FIELD_RX.items():
            if fname == "KEY":
                m = rx.search(chunk)
                if m:
                    # collect following bullet lines until blank or next header
                    rest = chunk[m.end():]
                    bullets = []
                    for line in rest.splitlines():
                        if re.match(r"^\s*$", line): break
                        if re.match(r"^\s*SCHOOL\s*:", line, flags=re.I): break
                        if re.match(r"^\s*[-*•]\s*(.+)$", line):
                            bullets.append(_norm_space(re.match(r"^\s*[-*•]\s*(.+)$", line).group(1)))
                        elif bullets and not line.strip().startswith(("ALIASES:","TRANSLATION:","TYPE:","FOCUS:","WEAPONS:")):
                            # continuation
                            bullets[-1] += " " + _norm_space(line)
                    data["KEY_POINTS"] = bullets
            else:
                m = rx.search(chunk)
                if m:
                    data[fname] = _norm_space(m.group(1))
        sections[key] = data
    return sections

def _compose_from_schema(key: str, sec: Dict[str, Any]) -> Optional[str]:
    pretty = _pretty(key)
    parts = [f"{pretty}: "]
    if "TRANSLATION" in sec:
        parts.append(f"Translation “{sec['TRANSLATION']}”. ")
    if "TYPE" in sec:
        parts.append(f"Type: {sec['TYPE']}. ")
    if "FOCUS" in sec:
        parts.append(f"Focus: {sec['FOCUS']}. ")
    if "WEAPONS" in sec and sec["WEAPONS"]:
        parts.append(f"Weapons: {sec['WEAPONS']}. ")
    if sec.get("KEY_POINTS"):
        # Add up to 2 crisp key points
        pts = sec["KEY_POINTS"][:2]
        parts.append("Key points: " + "; ".join(pts) + ".")
    text = "".join(parts).strip()
    return text if len(text) > len(pretty) + 3 else None

# ---------- Fallback slicer for unstructured prose ----------
def _slice_unstructured(blob: str, key: str) -> Optional[str]:
    if not blob: return None
    low = _strip_macrons(blob.lower())
    aliases = [_strip_macrons(a).lower() for a in SCHOOL_ALIASES.get(key, [])]
    starts = [low.find(a) for a in aliases if a in low]
    if not starts: return None
    start = min(i for i in starts if i >= 0)
    # stop at next blank line or next school alias
    other_aliases = []
    for k, als in SCHOOL_ALIASES.items():
        if k == key: continue
        other_aliases += [_strip_macrons(a).lower() for a in als]
    next_alias = len(blob)
    for a in other_aliases:
        j = low.find(a, start+1)
        if j != -1 and j < next_alias:
            next_alias = j
    m_break = re.search(r"\n\s*\n", blob[start:])
    next_break = start + m_break.start() if m_break else len(blob)
    end = min(next_alias, next_break, start + 1400)
    return blob[start:end].strip()

def try_answer_school_profile(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    key = _target_from_question(question)
    if not key:
        return None

    blob = _schools_blob(passages) or ""
    # 1) Structured path
    secs = _parse_schema_sections(blob)
    if key in secs:
        composed = _compose_from_schema(key, secs[key])
        if composed:
            return composed

    # 2) Fallback to unstructured slice + light scoring
    sec = _slice_unstructured(blob, key)
    if not sec:
        return None
    # pick sentences with concrete info
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9‘“])", _norm_space(sec))
    priority_words = ["ninpo","ninjutsu","taijutsu","kosshijutsu","koppojutsu","dakentaijutsu","jutaijutsu",
                      "stealth","infiltration","concealment","kamae","footwork","weapons","tools","kyoketsu","shuriken"]
    picked = [s for s in sents if any(w in _strip_macrons(s.lower()) for w in priority_words)]
    top = " ".join(picked[:3]) if picked else " ".join(sents[:3])
    pretty = _pretty(key)
    if not _strip_macrons(top.lower()).startswith(_strip_macrons(pretty.lower())):
        top = f"{pretty}: {top}"
    return top

def try_answer_schools(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    q = _strip_macrons(question.lower())
    if not (("school" in q) or ("schools" in q) or ("bujinkan" in q)):
        return None
    # Try to find compact list in context
    for p in passages:
        txt = p.get("text") or ""
        low = _strip_macrons(txt.lower())
        if all(any(a in low for a in aliases) for aliases in SCHOOL_ALIASES.values()):
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            joined = re.sub(r"\s+", " ", " ".join(lines))
            return joined[:400] + ("…" if len(joined) > 400 else "")
    return ("The Bujinkan encompasses nine classical ryū: Togakure-ryū, Gyokushin-ryū, Kumogakure-ryū, "
            "Gikan-ryū, Gyokko-ryū, Koto-ryū, Shinden Fudō-ryū, Kukishinden-ryū, and Takagi Yōshin-ryū.")
