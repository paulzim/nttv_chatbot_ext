# extractors/weapons.py
import re
from typing import List, Dict, Any, Optional

# ----------------------------
# Aliases & normalization
# ----------------------------
WEAPON_ALIASES = {
    "hanbo": ["hanbo", "hanbō", "short staff", "three-foot staff", "3-foot staff"],
    "rokushakubo": ["rokushakubo", "rokushaku bo", "rokushaku-bō", "bo", "long staff", "rokushaku"],
    "katana": ["katana", "sword", "daito", "daitō"],
    "tanto": ["tanto", "dagger", "knife"],
    "shoto": ["shoto", "short sword", "shōtō"],
    "kusari fundo": ["kusari fundo", "kusarifundo", "manriki-gusari", "manrikigusari", "weighted chain"],
    "naginata": ["naginata"],
    "kyoketsu shoge": ["kyoketsu shoge", "kyoketsu-shoge", "kyoketsu shōge"],
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

def _weapon_key_from_query(q: str) -> Optional[str]:
    ql = q.lower()
    for key, aliases in WEAPON_ALIASES.items():
        for a in aliases:
            if a in ql:
                return key
    # If they asked generic “weapons”, leave None → answer more carefully
    return None

# ----------------------------
# Parse [WEAPON] blocks from "NTTV Weapons Reference.txt"
# ----------------------------
def _parse_structured_weapons(passages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for p in passages:
        if "weapons reference" not in (p.get("source") or "").lower():
            continue
        text = p.get("text") or ""
        lines = text.splitlines()
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
            rows.append(current)
        # keep scanning in case file repeated
    return rows

def _find_structured_weapon(rows: List[Dict[str, str]], target_key: str) -> Optional[Dict[str, str]]:
    aliases = WEAPON_ALIASES.get(target_key, [])
    for r in rows:
        name_l = r.get("NAME", "").lower()
        if any(a in name_l for a in aliases):
            return r
        # also match if canonical key appears (loose)
        if target_key.replace("_", " ") in name_l:
            return r
    return None

# ----------------------------
# Rank → weapons mapping (auto-built from "nttv training reference.txt")
# ----------------------------
def _scan_rank_weapon_blocks_all(passages: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Build a dict like {"8th kyu": ["Hanbo", "Tanto"], ...} from nttv training reference.
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
            # detect rank header
            rh = RANK_HEAD.match(lines[i])
            if rh:
                current_rank = rh.group(1).title()  # normalize casing
                if current_rank not in out:
                    out[current_rank] = []
                i += 1
                continue
            # within a rank section, capture WEAPON(S)
            if current_rank:
                m = WEAPONS_BLOCK_HEAD.match(lines[i])
                if m:
                    # inline header text may already include some names
                    header_inline = _norm(m.group(1))
                    if header_inline:
                        out[current_rank].extend(_split_list(header_inline))
                    j = i + 1
                    # capture following bullets
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
    # tidy
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
    # Basic normalization for a few common forms
    normed = []
    for p in parts:
        low = p.lower()
        if "hanb" in low and "hanbo" not in low:
            normed.append("Hanbo"); continue
        if low in ("bo", "bō", "rokushaku", "rokushaku bo"):
            normed.append("Rokushakubo"); continue
        if "shoto" in low or "shōtō" in low:
            normed.append("Shoto"); continue
        if "tanto" in low or "knife" in low:
            normed.append("Tanto"); continue
        if "senban" in low or "shaken" in low or "shuriken" in low:
            normed.append("Shuriken"); continue
        normed.append(p.title())
    return normed

# ----------------------------
# Glossary fallback
# ----------------------------
def _glossary_one_liner(passages: List[Dict[str, Any]], target_key: str) -> Optional[str]:
    aliases = WEAPON_ALIASES.get(target_key, [])
    for p in passages:
        if "glossary" not in (p.get("source") or "").lower():
            continue
        for line in (p.get("text") or "").splitlines():
            ll = line.lower().strip()
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
# Public API
# ----------------------------
def try_answer_weapons(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic weapon explainer with structured fields support:
      1) Prefer [WEAPON] blocks in 'NTTV Weapons Reference.txt'
      2) Else summarize rank WEAPON(S) blocks
      3) Else glossary one-liner
      4) For shuriken, append a compact technique hint
    Also exposes auto-built rank→weapons mapping via try_build_rank_weapon_map(passages).
    """
    ql = question.lower()
    target = _weapon_key_from_query(ql)

    # (A) Structured file
    structured = _parse_structured_weapons(passages)
    if target and structured:
        hit = _find_structured_weapon(structured, target)
        if hit:
            # Build a compact, ordered sentence set
            parts = []
            name = hit.get("NAME", "").strip() or target.replace("_", " ").title()
            parts.append(name + ":")
            for field in FIELD_ORDER:
                if hit.get(field):
                    label = field.title().replace("/", " / ")
                    parts.append(f"{label}: {hit[field]}.")
            if target == "shuriken":
                parts.append(SHURIKEN_HINT)
            return " ".join(parts).strip()

    # (B) Rank WEAPONS block summary (works with or without target)
    rank_summary = _rank_weapons_summary(passages, target)
    if rank_summary:
        if target == "shuriken":
            return f"{rank_summary} {SHURIKEN_HINT}"
        return rank_summary

    # (C) Glossary fallback (targeted)
    if target:
        gloss = _glossary_one_liner(passages, target)
        if gloss:
            if target == "shuriken":
                return f"{gloss} {SHURIKEN_HINT}"
            return gloss

    # (D) Generic shuriken ask
    if any(a in ql for a in WEAPON_ALIASES["shuriken"]):
        return SHURIKEN_HINT

    return None

def _rank_weapons_summary(passages: List[Dict[str, Any]], target_key: Optional[str]) -> Optional[str]:
    """
    Produce a short sentence describing weapons per rank,
    filtered to the target if provided.
    """
    table = _scan_rank_weapon_blocks_all(passages)
    if not table:
        return None

    lines = []
    for rank in sorted(table.keys(), key=_rank_sort_key):
        weapons = table[rank]
        if target_key:
            aliases = WEAPON_ALIASES.get(target_key, [])
            sel = [w for w in weapons if any(a in w.lower() for a in aliases) or target_key.replace("_", " ") in w.lower()]
            if sel:
                lines.append(f"{rank}: {', '.join(sel[:6])}.")
        else:
            if weapons:
                lines.append(f"{rank}: {', '.join(weapons[:6])}.")
    if not lines:
        return None

    # Keep it tight: join up to ~3 ranks
    return " ".join(lines[:3])

def _rank_sort_key(rank: str) -> int:
    # Convert "8th Kyu" → 8 (lower is lower rank)
    m = re.search(r"(\d{1,2})", rank)
    if m:
        return int(m.group(1))
    return 99

# Exposed helper: build full rank→weapons map (pass the app’s CHUNKS if you want global)
def try_build_rank_weapon_map(passages_or_chunks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    return _scan_rank_weapon_blocks_all(passages_or_chunks)
