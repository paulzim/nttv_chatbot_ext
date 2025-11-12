# extractors/techniques.py
from __future__ import annotations
import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional

# Optional indexed helpers (we won't rely on them)
try:
    from .technique_loader import parse_technique_md, build_indexes
except Exception:
    parse_technique_md = None
    build_indexes = None

# Words that indicate concept questions, not single techniques
CONCEPT_BANS = ("kihon happo", "kihon happō", "sanshin", "school", "schools", "ryu", "ryū")

TRIGGERS = ("what is", "define", "explain")
NAME_HINTS = ("gyaku","dori","kudaki","gatame","otoshi","nage","seoi","kote",
              "musha","take ori","juji","omote","ura","ganseki","hodoki",
              "kata","no kata")

EXPECTED_COLS = 12  # positional schema (name,..., description absorbs commas)

def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _fold(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()

def _lite(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _fold(s))

def _looks_like_technique_q(q: str) -> bool:
    ql = _norm_space(q).lower()
    if any(b in ql for b in CONCEPT_BANS):
        return False
    if any(t in ql for t in TRIGGERS):
        return True
    if len(ql.split()) <= 7 and any(h in ql for h in NAME_HINTS):
        return True
    return False

def _gather_full_technique_text(passages: List[Dict[str, Any]]) -> str:
    buf = []
    for p in passages:
        src = (p.get("source") or "").lower()
        if "technique descriptions" in src:
            buf.append(p.get("text", ""))
    return "\n".join(buf)

def _extract_candidate(ql: str) -> str:
    m = re.search(r"(?:what\s+is|define|explain)\s+(.+)$", ql, flags=re.I)
    cand = (m.group(1) if m else ql).strip().rstrip("?!.")
    cand = re.sub(r"\b(technique|in ninjutsu|in bujinkan)\b", "", cand, flags=re.I).strip()
    return cand

def _candidate_variants(raw: str) -> List[str]:
    v: List[str] = []
    raw = _norm_space(raw)
    v.append(raw)
    # add/remove 'no kata'
    if raw.lower().endswith(" no kata"):
        v.append(raw[:-8].strip())
    else:
        v.append(f"{raw} no kata")
    # hyphen-insensitive + spacing
    raw_no_hy = raw.replace("-", " ")
    if raw_no_hy != raw:
        v.append(raw_no_hy)
        if raw_no_hy.lower().endswith(" no kata"):
            v.append(raw_no_hy[:-8].strip())
        else:
            v.append(f"{raw_no_hy} no kata")
    # folded + lite
    v.append(_fold(raw))
    v.append(_lite(raw))
    # de-dupe preserve order
    seen = set(); out = []
    for x in v:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def _fmt_bool(v: Optional[bool]) -> str:
    if v is True: return "Yes"
    if v is False: return "No"
    return "—"

def _format_bullets(rec: Dict[str, Any]) -> str:
    name = rec.get("name") or "Technique"
    translation = rec.get("translation") or ""
    typ = rec.get("type") or ""
    rank = rec.get("rank") or ""
    focus = rec.get("primary_focus") or ""
    safety = rec.get("safety") or rec.get("difficulty") or ""
    partner = _fmt_bool(rec.get("partner_required"))
    solo = _fmt_bool(rec.get("solo"))
    desc = (rec.get("description") or "").strip()

    lines = [f"{name}:"]
    if translation:   lines.append(f"- Translation: {translation}")
    if typ:           lines.append(f"- Type: {typ}")
    if rank:          lines.append(f"- Rank intro: {rank}")
    if focus:         lines.append(f"- Focus: {focus}")
    if safety:        lines.append(f"- Safety: {safety}")
    if partner != "—":lines.append(f"- Partner required: {partner}")
    if solo != "—":   lines.append(f"- Solo: {solo}")
    if desc:          lines.append(f"- Definition: {desc if desc.endswith('.') else desc + '.'}")
    else:             lines.append(f"- Definition: (not listed).")
    return "\n".join(lines)

# ----- CSV helpers (description absorbs commas) -----

def _iter_csv_like_lines(md_text: str):
    for raw in (md_text or "").splitlines():
        st = raw.strip()
        if not st:
            continue
        if st.startswith("#") or st.startswith("```"):
            continue
        if "," in raw:
            yield raw

def _split_row_limited(raw: str) -> List[str]:
    parts = raw.split(",", EXPECTED_COLS - 1)  # last field absorbs the rest
    parts = [p.strip() for p in parts]
    if len(parts) > EXPECTED_COLS:
        head = parts[:EXPECTED_COLS-1]
        tail = ",".join(parts[EXPECTED_COLS-1:])
        parts = head + [tail]
    if len(parts) < EXPECTED_COLS:
        parts += [""] * (EXPECTED_COLS - len(parts))
    return parts

def _scan_csv_rows_limited(md_text: str) -> List[List[str]]:
    return [_split_row_limited(raw) for raw in _iter_csv_like_lines(md_text)]

def _has_header(cells: List[str]) -> bool:
    header = [c.strip().lower() for c in cells]
    return any(h in {"name","translation","japanese","description"} for h in header)

def _to_bool(x: str) -> Optional[bool]:
    v = (x or "").strip().lower()
    if v in {"1","true","yes","y","✅","✓","✔"}: return True
    if v in {"0","false","no","n","❌","✗","✕"}: return False
    return None

def _row_to_record_positional(row: List[str]) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "name": row[0],
        "japanese": row[1],
        "translation": row[2],
        "type": row[3],
        "rank": row[4],
        "in_rank": row[5],
        "primary_focus": row[6],
        "safety": row[7],
        "partner_required": _to_bool(row[8]),
        "solo": _to_bool(row[9]),
        "tags": row[10],
        "description": row[11],
    }
    return rec

# ----- NEW: Direct line lookup anchored to technique name -----

def _direct_line_lookup(md_text: str, cand_variants: List[str]) -> Optional[Dict[str, Any]]:
    """
    Find a line that *starts* with the technique name (case/macrón-insensitive),
    then parse it as a positional CSV row.
    """
    # build set of folded anchors (e.g., 'jumonji no kata')
    anchors = { _fold(v) for v in cand_variants if v and not v.startswith("#") }
    if not anchors:
        return None

    for raw in (md_text or "").splitlines():
        line = raw.rstrip()
        if not line or "," not in line:
            continue
        # take the first field (before first comma), fold it, compare to anchors
        first = line.split(",", 1)[0].strip()
        if _fold(first) in anchors:
            row = _split_row_limited(line)
            return _row_to_record_positional(row)
    return None

# ----- CSV table lookup (fallback) -----

def _csv_fallback_lookup(md_text: str, cand_variants: List[str]) -> Optional[Dict[str, Any]]:
    rows = _scan_csv_rows_limited(md_text)
    if not rows:
        return None

    has_header = _has_header(rows[0])
    data_rows = rows[1:] if has_header else rows

    cand_folded = [_fold(c) for c in cand_variants]
    cand_lite = [_lite(c) for c in cand_variants]

    best = (None, 0.0, None)
    for r in data_rows:
        if not r or not r[0].strip():
            continue
        name = r[0].strip()
        name_fold = _fold(name)
        name_lite = _lite(name)

        if name_fold in cand_folded or name_lite in cand_lite:
            return _row_to_record_positional(r)

        score = max(SequenceMatcher(None, name_fold, cf).ratio() for cf in cand_folded)
        if score > best[1]:
            best = (r, score, name)

    if best[0] is not None and best[1] >= 0.85:
        return _row_to_record_positional(best[0])
    return None

# ----- Public entrypoint -----

def try_answer_technique(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    if not _looks_like_technique_q(question):
        return None

    md_text = _gather_full_technique_text(passages)
    if not md_text.strip():
        return None

    ql = _norm_space(question).lower()
    cand_raw = _extract_candidate(ql)
    variants = _candidate_variants(cand_raw)

    # 1) NEW: direct line lookup first (most robust for single-line CSV format)
    rec = _direct_line_lookup(md_text, variants)
    if rec:
        return _format_bullets(rec)

    # 2) CSV fallback (whole-file scan)
    rec = _csv_fallback_lookup(md_text, variants)
    if rec:
        return _format_bullets(rec)

    # 3) Optional indexed route if available
    if parse_technique_md and build_indexes:
        try:
            records = parse_technique_md(md_text)
            idx = build_indexes(records) if records else None
        except Exception:
            idx = None
        if idx:
            by_name = idx["by_name"]; by_lower = idx["by_lower"]
            by_fold = idx["by_fold"]; by_key = idx["by_keylite"]

            # direct keys
            for v in variants:
                key = v.lower()
                if key in by_lower:
                    return _format_bullets(by_name[by_lower[key]])
                fkey = _fold(v)
                if fkey in by_fold:
                    return _format_bullets(by_name[by_fold[fkey]])
                lkey = _lite(v)
                if lkey in by_key:
                    return _format_bullets(by_name[by_key[lkey]])

            # fuzzy
            cq = _fold(cand_raw)
            best_name, best_score = None, 0.0
            for name in by_name.keys():
                s = SequenceMatcher(None, cq, _fold(name)).ratio()
                if s > best_score:
                    best_name, best_score = name, s
            if best_name and best_score >= 0.80:
                return _format_bullets(by_name[best_name])

    return None
