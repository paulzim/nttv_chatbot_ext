# extractors/weapons.py
import re
from typing import List, Dict, Any, Optional

# ----------------------------
# Aliases & normalization  (preserved/extended)
# ----------------------------
WEAPON_ALIASES = {
    "hanbo": ["hanbo", "hanbō", "short staff", "three-foot staff", "3-foot staff"],
    "rokushakubo": ["rokushakubo", "rokushaku bo", "rokushaku-bō", "bo", "long staff", "rokushaku", "rokushaku staff"],
    "katana": ["katana", "sword", "daito", "daitō"],
    "tanto": ["tanto", "dagger", "knife"],
    "shoto": ["shoto", "short sword", "shōtō"],
    "kusari fundo": ["kusari fundo", "kusarifundo", "manriki-gusari", "manrikigusari", "weighted chain", "kusari-fundo"],
    "naginata": ["naginata"],
    "kyoketsu shoge": ["kyoketsu shoge", "kyoketsu-shoge", "kyoketsu shōge", "kyoketsu-shōge"],
    "shuko": ["shuko", "shukō", "hand claws"],
    "jutte": ["jutte", "jitte", "sai"],
    "tessen": ["tessen", "iron fan"],
    "kunai": ["kunai", "utility blade", "digging knife"],
    "shuriken": ["shuriken", "bo shuriken", "bō shuriken", "senban", "shaken", "throwing star", "throwing spike"],
}

FIELD_ORDER = [
    "TYPE", "KAMAE", "CORE ACTIONS", "MODES", "THROWS",
    "DISTANCE", "ANGLES", "TARGETS", "RANGE", "RANKS", "SAFETY/DRILL", "NOTES"
]

SHURIKEN_HINT = (
    "Shuriken has two main uses: daken-jutsu (throwing) and shoken-jutsu (in-hand). "
    "Common throws include jikidaho (direct), hantendaho (reverse), and takai-tendaho (fast spin)."
)

WEAPON_HEADER = re.compile(r"^\s*\[WEAPON\]\s*(.+?)\s*$", re.IGNORECASE)
FIELD_LINE = re.compile(
    r"^\s*(ALIASES|TYPE|KAMAE|CORE ACTIONS|MODES|THROWS|DISTANCE|ANGLES|TARGETS|RANGE|RANKS|SAFETY/DRILL|NOTES)\s*:\s*(.+?)\s*$",
    re.IGNORECASE
)

RANK_HEAD = re.compile(r"^\s*(\d{1,2}(?:st|nd|rd|th)\s+kyu)\b", re.IGNORECASE)
WEAPONS_BLOCK_HEAD = re.compile(r"^\s*WEAPONS?\s*:\s*(.+?)\s*$", re.IGNORECASE)
BULLET = re.compile(r"^\s*[•\-\u2022]\s*(.+?)\s*$")

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _strip_macrons(s: str) -> str:
    return (s or "").translate(str.maketrans({
        "ō":"o","ū":"u","ā":"a","ī":"i","Ō":"O","Ū":"U","Ā":"A","Ī":"I"
    }))

def _weapon_key_from_query(q: str) -> Optional[str]:
    ql = _strip_macrons(q.lower())
    for key, aliases in WEAPON_ALIASES.items():
        for a in aliases:
            if _strip_macrons(a).lower() in ql:
                return key
    return None

# ----------------------------
# Parse [WEAPON] blocks from "NTTV Weapons Reference.txt"
# ----------------------------
def _extract_weapon_structured_blob(passages: List[Dict[str, Any]]) -> str:
    parts = []
    for p in passages:
        src = (p.get("source") or "").lower()
        if "weapons reference" in src:
            parts.append(p.get("text") or "")
    return "\n".join(parts)

def _parse_structured_weapons(blob: str) -> List[Dict[str, str]]:
    """
    Parse all [WEAPON] ... blocks into a list of dicts with labeled fields.
    """
    rows: List[Dict[str, str]] = []
    if not blob:
        return rows

    lines = blob.splitlines()
    i = 0
    while i < len(lines):
        m = WEAPON_HEADER.match(lines[i])
        if not m:
            i += 1
            continue
        current: Dict[str, str] = {"NAME": _norm(m.group(1))}
        i += 1
        while i < len(lines):
            ln = lines[i].strip()
            if not ln:
                break
            if WEAPON_HEADER.match(ln):
                break
            f = FIELD_LINE.match(lines[i])
            if f:
                field = f.group(1).upper()
                val = _norm(f.group(2))
                current[field] = val
            i += 1
        # expand aliases from the field, but preserve global table for matching
        if "ALIASES" in current:
            current["ALIASES"] = ", ".join([a.strip() for a in current["ALIASES"].split(",") if a.strip()])
        rows.append(current)
    return rows

def _find_structured_weapon(rows: List[Dict[str, str]], target_key: str) -> Optional[Dict[str, str]]:
    aliases = [a.lower() for a in WEAPON_ALIASES.get(target_key, [])]
    for r in rows:
        name_l = _strip_macrons(r.get("NAME", "")).lower()
        if any(a in name_l for a in aliases):
            return r
        if target_key.replace("_", " ") in name_l:
            return r
    return None

# ----------------------------
# Rank → weapons mapping (auto-built from "nttv training reference.txt")
# ----------------------------
def _scan_rank_weapon_blocks_all(passages: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Build a dict like {"8th Kyu": ["Hanbo", "Tanto"], ...} from nttv training reference.
    Looks for rank headers followed by WEAPON(S) lines and bullets.
    """
    out: Dict[str, List[str]] = {}
    for p in passages:
        if "nttv training reference" not in (p.get("source") or "").lower():
            continue
        lines = (p.get("text") or "").splitlines()
        current_rank = None
        i = 0
        while i < len(lines):
            rh = RANK_HEAD.match(lines[i])
            if rh:
                current_rank = rh.group(1).title()
                if current_rank not in out:
                    out[current_rank] = []
                i += 1
                continue
            if current_rank:
                m = WEAPONS_BLOCK_HEAD.match(lines[i])
                if m:
                    header_inline = _norm(m.group(1))
                    if header_inline:
                        out[current_rank].extend(_split_list(header_inline))
                    j = i + 1
                    while j < len(lines):
                        ln = lines[j].strip()
                        if not ln:
                            break
                        if WEAPONS_BLOCK_HEAD.match(ln) or RANK_HEAD.match(ln):
                            break
                        b = BULLET.match(ln)
                        if b:
                            out[current_rank].extend(_split_list(_norm(b.group(1))))
                        j += 1
                    i = j
                    continue
            i += 1
    # de-dup & tidy
    for rk, vals in out.items():
        seen, clean = set(), []
        for v in vals:
            vv = _norm(v)
            if vv and vv.lower() not in seen:
                seen.add(vv.lower()); clean.append(vv)
        out[rk] = clean
    return out

def _split_list(s: str) -> List[str]:
    parts = [p.strip(" \t-•") for p in re.split(r"[;,/]| and ", s) if p.strip()]
    normed = []
    for p in parts:
        low = _strip_macrons(p.lower())
        if "hanb" in low and "hanbo" not in low:
            normed.append("Hanbo"); continue
        if low in ("bo", "bō", "rokushaku", "rokushaku bo", "rokushaku staff"):
            normed.append("Rokushakubo"); continue
        if "shoto" in low or "shōtō" in low:
            normed.append("Shoto"); continue
        if "tanto" in low or "knife" in low:
            normed.append("Tanto"); continue
        if "senban" in low or "shaken" in low or "shuriken" in low or "throwing star" in low or "throwing spike" in low:
            normed.append("Shuriken"); continue
        if "kusari" in low and "fundo" in low:
            normed.append("Kusari Fundo"); continue
        if "kyoketsu" in low and "shoge" in low:
            normed.append("Kyoketsu Shoge"); continue
        normed.append(p.title())
    return normed

# ----------------------------
# Glossary fallback
# ----------------------------
def _glossary_one_liner(passages: List[Dict[str, Any]], target_key: str) -> Optional[str]:
    aliases = [a.lower() for a in WEAPON_ALIASES.get(target_key, [])]
    for p in passages:
        if "glossary" not in (p.get("source") or "").lower():
            continue
        for line in (p.get("text") or "").splitlines():
            ll = _strip_macrons(line.lower().strip())
            if any(a in ll for a in aliases):
                clean = _norm(line)
                # "Term - definition" or "Term – definition"
                if " - " in clean:
                    term, rest = clean.split(" - ", 1)
                    if len(rest) > 2:
                        return f"{term.strip()}: {rest.strip()}."
                if " – " in clean:
                    term, rest = clean.split(" – ", 1)
                    if len(rest) > 2:
                        return f"{term.strip()}: {rest.strip()}."
    return None

# ----------------------------
# Public deterministic answers
# ----------------------------
def try_answer_weapon_rank(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Answer: 'Kusari Fundo is introduced at 4th Kyu.' (or nearest equivalent)
    Preference order:
      1) RANKS field in [WEAPON] block from NTTV Weapons Reference.txt
      2) First rank in WEAPONS list for that weapon from nttv training reference.txt
      3) Glossary (if at least defines, but then no rank will be stated)
    """
    target = _weapon_key_from_query(question)
    if not target:
        return None

    # (1) Structured reference
    blob = _extract_weapon_structured_blob(passages)
    rows = _parse_structured_weapons(blob)
    if rows:
        hit = _find_structured_weapon(rows, target)
        if hit and hit.get("RANKS"):
            name = hit.get("NAME", target.title())
            ranks = hit["RANKS"]
            # Normalize phrasing a bit
            intro = ranks
            if any(k in ranks.lower() for k in ["introduced at", "learn at", "start at"]):
                intro = ranks
            else:
                intro = f"Introduced at {ranks}"
            return f"{name} is {intro}."

    # (2) Rank WEAPONS map
    table = _scan_rank_weapon_blocks_all(passages)
    if table:
        first_rank = _first_rank_for_weapon(table, target)
        if first_rank:
            return f"{_display_name_for_target(rows, target)} is introduced at {first_rank}."

    # (3) Glossary fallback (definition only)
    gloss = _glossary_one_liner(passages, target)
    if gloss:
        return gloss

    return None

def try_answer_weapon_profile(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Short profile: 'Kusari Fundo: TYPE … Core actions include …'
    """
    target = _weapon_key_from_query(question)
    if not target:
        return None

    # Prefer structured reference
    blob = _extract_weapon_structured_blob(passages)
    rows = _parse_structured_weapons(blob)
    if rows:
        hit = _find_structured_weapon(rows, target)
        if hit:
            name = hit.get("NAME", target.title())
            parts = [f"{name}"]
            if hit.get("TYPE"):
                parts.append(f": {hit['TYPE']}")
            # Include a few key fields
            for field in ["CORE ACTIONS", "DISTANCE", "RANGE", "TARGETS", "NOTES"]:
                if hit.get(field):
                    label = field.title().replace("/", " / ")
                    if field == "CORE ACTIONS":
                        parts.append(f". Core actions include {hit[field].lower()}")
                    else:
                        parts.append(f". {label}: {hit[field]}")
            return "".join(parts).strip() + "."

    # Glossary definition as last resort
    gloss = _glossary_one_liner(passages, target)
    if gloss:
        return gloss

    return None

# ----------------------------
# Helpers
# ----------------------------
def _rank_sort_key(rank: str) -> int:
    m = re.search(r"(\d{1,2})", rank)
    return int(m.group(1)) if m else 99

def _first_rank_for_weapon(table: Dict[str, List[str]], target_key: str) -> Optional[str]:
    # Find the lowest-numbered kyu that lists the weapon (by alias)
    aliases = [a.lower() for a in WEAPON_ALIASES.get(target_key, [])]
    best = None
    for rank in sorted(table.keys(), key=_rank_sort_key):
        weapons = table[rank]
        for w in weapons:
            wl = _strip_macrons(w.lower())
            if any(a in wl for a in aliases) or target_key.replace("_", " ") in wl:
                best = rank
                break
        if best:
            break
    return best

def _display_name_for_target(rows: List[Dict[str, str]], target_key: str) -> str:
    if rows:
        # Show the first structured NAME that matches the target
        hit = _find_structured_weapon(rows, target_key)
        if hit and hit.get("NAME"):
            return hit["NAME"]
    # Fallback: title-case canonical key
    return target_key.title().replace("_", " ")
