# extractors/schools.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import re

# ----------------------------
# Canonical names + aliases
# ----------------------------
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

# ----------------------------
# Normalization helpers
# ----------------------------
_MACRON_MAP = str.maketrans({
    "ō": "o", "Ō": "O",
    "ū": "u", "Ū": "U",
    "ā": "a", "Ā": "A",
    "ī": "i", "Ī": "I",
    "ē": "e", "Ē": "E",
    "’": "'", "“": '"', "”": '"',
})

def _norm(s: str) -> str:
    s = (s or "").translate(_MACRON_MAP)
    s = s.replace("\u2010", "-").replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def _looks_like_school_header(line: str) -> bool:
    t = _norm(line)
    return (
        t.startswith("school:") or
        t.startswith("school -") or
        t.startswith("school –") or
        t.endswith(" ryu:") or
        t.endswith(" ryu :")
    )

def _canon_for_query(question: str) -> Optional[str]:
    qn = _norm(question)
    for canon, aliases in SCHOOL_ALIASES.items():
        tokens = [_norm(canon)] + [_norm(a) for a in aliases]
        if any(tok in qn for tok in tokens):
            return canon
    m = re.search(r"([a-z0-9\- ]+)\s+ryu\b", qn)
    if m:
        guess = m.group(1).strip().replace("-", " ")
        for canon in SCHOOL_ALIASES.keys():
            if _norm(canon).startswith(guess):
                return canon
    return None

# ----------------------------
# List-intent detection (EXPORTED)
# ----------------------------
def is_school_list_query(question: str) -> bool:
    q = _norm(question)
    triggers = [
        "what are the schools of the bujinkan",
        "list the schools of the bujinkan",
        "nine schools of the bujinkan",
        "what are the nine schools",
        "list the nine schools",
        "what schools are in the bujinkan",
        "which schools are in the bujinkan",
    ]
    return any(t in q for t in triggers)

# ----------------------------
# Slicing & field extraction
# ----------------------------
_FIELD_KEYS = ["translation", "type", "focus", "weapons", "notes"]

def _slice_school_blocks(blob: str) -> List[Tuple[str, List[str]]]:
    lines = blob.splitlines()
    idxs = [i for i, ln in enumerate(lines) if _looks_like_school_header(ln)]
    blocks: List[Tuple[str, List[str]]] = []
    for j, i in enumerate(idxs):
        start = i
        end = idxs[j + 1] if j + 1 < len(idxs) else len(lines)
        for k in range(i + 1, end):
            if lines[k].strip() == "---":
                end = k
                break
        blocks.append((lines[start], lines[start + 1:end]))
    return blocks

def _header_matches(header_line: str, canon: str) -> bool:
    h = _norm(header_line)
    cn = _norm(canon)
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
            data[key] = (data.get(key, "") + (" " if key in data and data[key] else "") + val).strip()
        else:
            if data:
                last_key = list(data.keys())[-1]
                data[last_key] = (data[last_key] + " " + ln.strip()).strip()
    return {k: v.strip() for k, v in data.items() if k in _FIELD_KEYS and v.strip()}

def _format_profile(canon: str, fields: Dict[str, str], bullets: bool) -> str:
    title = canon
    if bullets:
        parts = [f"{title}:"]
        for k in ["translation", "type", "focus", "weapons", "notes"]:
            if k in fields:
                parts.append(f"- {k.capitalize()}: {fields[k]}")
        return "\n".join(parts)
    segs = []
    if "translation" in fields:
        segs.append(f'“{fields["translation"]}”.')
    if "type" in fields:
        segs.append(f'Type: {fields["type"]}.')
    if "focus" in fields:
        segs.append(f'Focus: {fields["focus"]}.')
    if "weapons" in fields:
        segs.append(f'Weapons: {fields["weapons"]}.')
    if "notes" in fields:
        segs.append(f'Notes: {fields["notes"]}.')
    return f"{title}: " + (" ".join(segs) if segs else "")

def _collect_schools_blob(passages: List[Dict[str, Any]]) -> str:
    candidates: List[Tuple[int, int, str]] = []  # (syn_flag, -len, text)
    for p in passages:
        src = (p.get("source") or "").lower()
        txt = (p.get("text") or "").strip()
        if not txt:
            continue
        if "schools of the bujinkan summaries" in src:
            syn = 0 if "(synthetic)" in src else 1
            candidates.append((syn, -len(txt), txt))
        else:
            if "school:" in _norm(txt):
                candidates.append((1, -len(txt), txt))
    if not candidates:
        return ""
    candidates.sort()
    seen = set()
    out: List[str] = []
    for _, _, t in candidates:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return "\n\n".join(out)

def _fallback_block_by_alias(blob: str, canon: str) -> Optional[List[str]]:
    if not blob.strip():
        return None
    lines = blob.splitlines()
    norm_lines = [_norm(ln) for ln in lines]
    aliases = [_norm(canon)] + [_norm(a) for a in SCHOOL_ALIASES.get(canon, [])]
    hit_idx = None
    for i, ln in enumerate(norm_lines):
        if any(tok in ln for tok in aliases):
            hit_idx = i
            break
    if hit_idx is None:
        return None
    start = max(0, hit_idx - 3)
    end = min(len(lines), hit_idx + 25)
    for j in range(hit_idx + 1, end):
        if lines[j].strip() == "---" or _looks_like_school_header(lines[j]):
            end = j
            break
    return lines[start:end]

def _infer_fields_from_freeblock(free_lines: List[str]) -> Dict[str, str]:
    txt = "\n".join(free_lines)
    data = _parse_fields(free_lines)
    if data:
        return data
    n = _norm(txt)
    inferred: Dict[str, str] = {}
    if any(k in n for k in ["ninpo", "ninjutsu"]):
        inferred["type"] = "Ninjutsu"
    elif any(k in n for k in ["kosshijutsu", "koppojutsu", "dakentaijutsu", "jutaijutsu", "samurai"]):
        inferred["type"] = "Samurai"
    m = re.search(r'translation[: ]+["“](.+?)["”]', txt, flags=re.IGNORECASE)
    if m:
        inferred["translation"] = m.group(1).strip()
    focus_terms = []
    for term in ["stealth", "infiltration", "surprise", "espionage", "distance", "timing", "kamae",
                 "kosshijutsu", "koppojutsu", "striking", "bone", "joint", "throws", "grappling",
                 "dakentaijutsu", "jutaijutsu"]:
        if term in n:
            focus_terms.append(term)
    if focus_terms:
        inferred["focus"] = ", ".join(sorted(set(focus_terms)))
    wterms = []
    for term in ["shuriken", "kunai", "kodachi", "katana", "yari", "naginata", "bo", "hanbo",
                 "kusarifundo", "kusari fundo", "kyoketsu shoge", "tessen", "jutte", "jitte"]:
        if term in n:
            wterms.append(term)
    if wterms:
        inferred["weapons"] = ", ".join(sorted(set(wterms)))
    return inferred

def _canon_from_header(header_line: str) -> Optional[str]:
    h = _norm(header_line)
    for canon, aliases in SCHOOL_ALIASES.items():
        tokens = [_norm(canon)] + [_norm(a) for a in aliases]
        if any(tok in h for tok in tokens):
            return canon
    m = re.search(r"([a-z0-9\- ]+)\s+ryu\b", h)
    if m:
        guess = m.group(1).strip().replace("-", " ")
        for canon in SCHOOL_ALIASES.keys():
            if _norm(canon).startswith(guess):
                return canon
    return None

# ----------------------------
# Public API (EXPORTED)
# ----------------------------
def try_answer_schools_list(
    question: str,
    passages: List[Dict[str, Any]],
    *,
    bullets: bool = True,
) -> Optional[str]:
    if not is_school_list_query(question):
        return None
    blob = _collect_schools_blob(passages)
    if not blob.strip():
        return None
    blocks = _slice_school_blocks(blob)
    if not blocks:
        return None
    names: List[str] = []
    seen = set()
    for header, _ in blocks:
        canon = _canon_from_header(header)
        if canon and canon not in seen:
            seen.add(canon)
            names.append(canon)
    if not names:
        return None
    canonical_order = [
        "Togakure Ryu",
        "Gyokushin Ryu",
        "Kumogakure Ryu",
        "Gikan Ryu",
        "Gyokko Ryu",
        "Koto Ryu",
        "Shinden Fudo Ryu",
        "Kukishinden Ryu",
        "Takagi Yoshin Ryu",
    ]
    if set(n.lower() for n in names) >= set(n.lower() for n in canonical_order):
        order_map = {n.lower(): i for i, n in enumerate(canonical_order)}
        names.sort(key=lambda s: order_map.get(s.lower(), 999))
    title = "The Nine Schools of the Bujinkan"
    if bullets:
        out = [f"{title}:"]
        for n in names:
            out.append(f"- {n}")
        return "\n".join(out)
    return f"{title}: " + ", ".join(names) + "."

def try_answer_school_profile(
    question: str,
    passages: List[Dict[str, Any]],
    *,
    bullets: bool = True,
) -> Optional[str]:
    canon = _canon_for_query(question)
    if not canon:
        return None
    blob = _collect_schools_blob(passages)
    if not blob.strip():
        return None
    blocks = _slice_school_blocks(blob)
    fields: Optional[Dict[str, str]] = None
    if blocks:
        for header, body in blocks:
            if _header_matches(header, canon):
                fields = _parse_fields(body)
                if fields:
                    break
        if not fields:
            cn = _norm(canon)
            for header, body in blocks:
                all_text = _norm(" ".join([header] + body))
                if cn in all_text:
                    fields = _parse_fields(body)
                    if fields:
                        break
    if not fields:
        window = _fallback_block_by_alias(blob, canon)
        if window:
            fields = _infer_fields_from_freeblock(window)
    if not fields:
        return None
    return _format_profile(canon, fields, bullets=bullets)
