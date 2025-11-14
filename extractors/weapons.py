# extractors/weapons.py
import re
from typing import List, Dict, Any, Optional

# ----------------------------
# Aliases & normalization  (preserved/extended)
# ----------------------------
WEAPON_ALIASES = {
    "hanbo": ["hanbo", "hanbō", "short staff", "three-foot staff", "3-foot staff"],
    "rokushakubo": [
        "rokushakubo",
        "rokushaku bo",
        "rokushaku-bō",
        "bo",
        "long staff",
        "rokushaku",
        "rokushaku staff",
    ],
    "katana": ["katana", "sword", "daito", "daitō"],
    "tanto": ["tanto", "dagger", "knife"],
    "shoto": ["shoto", "short sword", "shōtō"],
    "kusari fundo": [
        "kusari fundo",
        "kusarifundo",
        "manriki-gusari",
        "manrikigusari",
        "weighted chain",
        "kusari-fundo",
    ],
    "naginata": ["naginata"],
    "kyoketsu shoge": [
        "kyoketsu shoge",
        "kyoketsu-shoge",
        "kyoketsu shōge",
        "kyoketsu-shōge",
    ],
    "shuriken": [
        "shuriken",
        "bo shuriken",
        "bō shuriken",
        "senban",
        "shaken",
        "throwing star",
        "throwing stars",
        "throwing spike",
        "throwing spikes",
    ],
}

SHURIKEN_FALLBACK_PROFILE = (
    "Shuriken: Throwing blades and spikes used for distraction, pain compliance, and precise targeting. "
    "Modes include daken-jutsu (throwing) and shoken-jutsu (in-hand striking). "
    "Common throws include jikidaho (direct), hantendaho (reverse), and takai-tendaho (fast spin)."
)

WEAPON_HEADER = re.compile(r"^\s*\[WEAPON\]\s*(.+?)\s*$", re.IGNORECASE)
FIELD_LINE = re.compile(
    r"^\s*(ALIASES|TYPE|KAMAE|CORE ACTIONS|MODES|THROWS|DISTANCE|ANGLES|TARGETS|RANGE|RANKS|SAFETY/DRILL|NOTES)\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)

RANK_HEAD = re.compile(r"^\s*(\d{1,2}(?:st|nd|rd|th)\s+kyu)\b", re.IGNORECASE)
WEAPONS_BLOCK_HEAD = re.compile(r"^\s*WEAPONS?\s*:\s*(.+?)\s*$", re.IGNORECASE)
BULLET = re.compile(r"^\s*[•\-\u2022]\s*(.+?)\s*$")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _strip_macrons(s: str) -> str:
    return (s or "").translate(
        str.maketrans(
            {
                "ō": "o",
                "ū": "u",
                "ā": "a",
                "ī": "i",
                "Ō": "O",
                "Ū": "U",
                "Ā": "A",
                "Ī": "I",
            }
        )
    )


def _weapon_key_from_query(q: str) -> Optional[str]:
    ql = _strip_macrons(q.lower())
    for key, aliases in WEAPON_ALIASES.items():
        for a in aliases:
            if _strip_macrons(a).lower() in ql:
                return key
    return None


# ----------------------------
# Shuriken types & katana parts helpers
# ----------------------------

SHURIKEN_TYPES_TEXT = (
    "Major types of shuriken in this curriculum include:\n"
    "• Bo-shuriken (straight spike types) — like needles, darts, or nails, often square, round, or octagonal in cross-section, "
    "thrown in a direct or single-spin style.\n"
    "• Hira-shuriken (flat 'ninja stars') — multi-point, bladed forms thrown with a slicing rotation at short ranges.\n"
    "• Senban shuriken — a specific four-pointed, square-like hira-shuriken, historically prominent in Togakure-ryū and useful "
    "both as a thrown weapon and as a ground trap.\n"
    "• Needle / hari-gata shuriken — very thin, needle-like forms suited to precise, stealthy insertion and often associated "
    "with poison use.\n"
    "• Modern improvised throwing tools — chopsticks, pens, nails, allen wrenches, hex keys, and similar objects that can be "
    "trained as shuriken as long as they fit the hand, are controllable, and can pierce, distract, or strike.\n"
)

# Canonical katana part names we care about; descriptions are taken from the
# 'Terminology' block in nttv training reference.txt at runtime.
KATANA_PART_NAMES = {
    "tsuka",
    "tsuka kishiri",
    "saya",
    "sageo",
    "tsuba",
    "ha",
    "hi",
    "hamon",
    "mune",
    "kissaki",
}


def _wants_weapon_rank(question: str) -> bool:
    """Return True only if the question clearly looks like a rank/when-do-I-learn query."""
    q = _strip_macrons((question or "").lower())

    if "what rank" in q or "which rank" in q or "at what rank" in q:
        return True
    if "rank do i" in q or "rank do we" in q:
        return True

    if "kyu" in q or "dan" in q or "shodan" in q:
        return True

    if "when do i learn" in q or "when do we learn" in q:
        return True
    if "what level" in q and any(w in q for w in ["learn", "start", "introduced"]):
        return True

    return False


def _wants_shuriken_types(question: str) -> bool:
    q = _strip_macrons((question or "").lower())
    if "shuriken" not in q:
        return False
    return any(
        w in q for w in ["type", "types", "kind", "kinds", "forms", "classification", "categories"]
    )


def _wants_katana_parts(question: str) -> bool:
    q = _strip_macrons((question or "").lower())
    if "katana" not in q and "sword" not in q:
        return False
    if "parts" in q or "terminology" in q:
        return True
    if "name the parts" in q or "what are the parts" in q:
        return True
    return False


def _extract_katana_parts(passages: List[Dict[str, Any]]) -> List[str]:
    """Pull katana part lines from the 'Terminology' section of nttv training reference."""
    parts: List[str] = []
    for p in passages:
        src = (p.get("source") or "").lower()
        if "nttv training reference" not in src:
            continue
        text = p.get("text") or ""
        for raw in text.splitlines():
            line = raw.strip()
            if not line.startswith("·"):
                continue
            body = line.lstrip("·").strip()
            if "-" not in body:
                continue
            name, desc = body.split("-", 1)
            name_clean = name.strip()
            key = _strip_macrons(name_clean.lower())
            if key in KATANA_PART_NAMES:
                parts.append(f"{name_clean}: {desc.strip()}")
    # De-duplicate while preserving order
    seen = set()
    ordered: List[str] = []
    for entry in parts:
        key = entry.split(":", 1)[0].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(entry)
    return ordered


def try_answer_katana_parts(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """Deterministic answer listing the parts of the katana from the curriculum text."""
    if not _wants_katana_parts(question):
        return None
    parts = _extract_katana_parts(passages)
    if not parts:
        return None
    bullet_lines = "\n".join(f"• {p}" for p in parts)
    return "Parts of the katana in this curriculum:\n" + bullet_lines + "\n"


# ----------------------------
# Parse [WEAPON] blocks from "NTTV Weapons Reference.txt"
# ----------------------------
def _extract_weapon_structured_blob(passages: List[Dict[str, Any]]) -> str:
    blob_parts: List[str] = []
    for p in passages:
        src = (p.get("source") or "").lower()
        if "weapons reference" not in src:
            continue
        blob_parts.append(p.get("text") or "")
    return "\n".join(blob_parts)


def _parse_structured_weapons(blob: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not blob:
        return rows

    lines = blob.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        m = WEAPON_HEADER.match(lines[i])
        if not m:
            i += 1
            continue
        name = _norm(m.group(1))
        current: Dict[str, str] = {"NAME": name}
        i += 1
        # read until blank line or next header
        while i < n:
            ln = lines[i]
            if not ln.strip():
                i += 1
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
            current["ALIASES"] = ", ".join(
                [a.strip() for a in current["ALIASES"].split(",") if a.strip()]
            )
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

    We look for blocks of the form:

        8th KYU
        ...
        Weapon: Hanbo
        WEAPONS: Hanbo, Tanto

    or

        WEAPONS:
        · Hanbo
        · Tanto
    """
    table: Dict[str, List[str]] = {}
    for p in passages:
        src = (p.get("source") or "").lower()
        if "nttv training reference" not in src:
            continue
        text = p.get("text") or ""
        lines = text.splitlines()
        current_rank: Optional[str] = None
        in_weapons = False
        for raw in lines:
            line = raw.rstrip()
            m_rank = RANK_HEAD.match(line)
            if m_rank:
                current_rank = m_rank.group(1).title()
                in_weapons = False
                continue
            if current_rank is None:
                continue
            if WEAPONS_BLOCK_HEAD.match(line):
                in_weapons = True
                if ":" in line:
                    after = line.split(":", 1)[1]
                    if after.strip():
                        items = [x.strip() for x in after.split(",") if x.strip()]
                        table.setdefault(current_rank, []).extend(items)
                continue
            if in_weapons:
                if not line.strip():
                    in_weapons = False
                    continue
                b = BULLET.match(line)
                if b:
                    item = b.group(1).strip()
                    if item:
                        table.setdefault(current_rank, []).append(item)
    # Normalize whitespace
    for k, v in list(table.items()):
        table[k] = [_norm(x) for x in v]
    return table


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

    # If the user is clearly asking about kamae / stances, let the Kamae
    # extractor handle it instead of treating this as a rank question.
    q_low = (question or "").lower()
    if "kamae" in q_low or "stance" in q_low or "stances" in q_low:
        return None

    target = _weapon_key_from_query(question)
    if not target:
        return None

    # Only handle true rank / when-do-I-learn questions here.
    if not _wants_weapon_rank(question):
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
            name = _display_name_for_target(rows, target)
            return f"{name} is introduced at {first_rank}."

    # (3) Glossary fallback (no rank info, but at least define the weapon)
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

    # Special handling: shuriken type/classification questions
    if target == "shuriken" and _wants_shuriken_types(question):
        return SHURIKEN_TYPES_TEXT

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

    # If we still have nothing and the user asked about shuriken, use a safe
    # fallback summary so they get something useful.
    if target == "shuriken":
        return SHURIKEN_FALLBACK_PROFILE

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
