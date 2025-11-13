# extractors/techniques.py
from __future__ import annotations
import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional

# Optional indexed helpers (kept — only used if available)
try:
    from .technique_loader import parse_technique_md, build_indexes
except Exception:
    parse_technique_md = None
    build_indexes = None

# Questions that should NOT be treated as single-technique lookups
CONCEPT_BANS = ("kihon happo", "kihon happō", "sanshin", "school", "schools", "ryu", "ryū")

# Light heuristics to spot single-technique queries
TRIGGERS = ("what is", "define", "explain")
NAME_HINTS = (
    "gyaku","dori","kudaki","gatame","otoshi","nage","seoi","kote",
    "musha","take ori","juji","omote","ura","ganseki","hodoki",
    "kata","no kata"
)

# Our CSV-like single-line schema (last field absorbs commas)
EXPECTED_COLS = 12  # name, japanese, translation, type, rank, in_rank, primary_focus,
                    # safety, partner_required, solo, tags, description

# ------------------------- utilities -------------------------

def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _fold(s: str) -> str:
    """Lowercase + strip macrons/diacritics."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()

def _lite(s: str) -> str:
    """Alnum only for tolerant matching."""
    return re.sub(r"[^a-z0-9]+", "", _fold(s))

def _looks_like_technique_q(q: str) -> bool:
    ql = _norm_space(q).lower()
    if any(b in ql for b in CONCEPT_BANS):
        return False
    if any(t in ql for t in TRIGGERS):
        return True
    # short, name-like things (e.g., “omote gyaku”, “jumonji no kata”)
    if len(ql.split()) <= 7 and any(h in ql for h in NAME_HINTS):
        return True
    return False

def _gather_full_technique_text(passages: List[Dict[str, Any]]) -> str:
    """Concatenate only Technique Descriptions docs."""
    buf = []
    for p in passages:
        src = (p.get("source") or "").lower()
        if "technique descriptions" in src:
            buf.append(p.get("text", ""))
    return "\n".join(buf)

def _extract_candidate(ql: str) -> str:
    """Pull the thing after 'what is|define|explain' … else return ql."""
    m = re.search(r"(?:what\s+is|define|explain)\s+(.+)$", ql, flags=re.I)
    cand = (m.group(1) if m else ql).strip().rstrip("?!.")
    cand = re.sub(r"\b(technique|in ninjutsu|in bujinkan)\b", "", cand, flags=re.I).strip()
    return cand

def _candidate_variants(raw: str) -> List[str]:
    """Generate robust name variants: +/- 'no kata', hyphen/space, folded/lite."""
    v: List[str] = []
    raw = _norm_space(raw)
    v.append(raw)

    # +/- 'no kata'
    if raw.lower().endswith(" no kata"):
        v.append(raw[:-8].strip())
    else:
        v.append(f"{raw} no kata")

    # Hyphen-insensitive
    raw_no_hy = raw.replace("-", " ")
    if raw_no_hy != raw:
        v.append(raw_no_hy)
        if raw_no_hy.lower().endswith(" no kata"):
            v.append(raw_no_hy[:-8].strip())
        else:
            v.append(f"{raw_no_hy} no kata")

    # Folded and “lite” forms
    v.append(_fold(raw))
    v.append(_lite(raw))

    # De-duplicate, preserve order
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
    """Consistent, compact bullet formatting."""
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
    if translation:     lines.append(f"- Translation: {translation}")
    if typ:             lines.append(f"- Type: {typ}")
    if rank:            lines.append(f"- Rank intro: {rank}")
    if focus:           lines.append(f"- Focus: {focus}")
    if safety:          lines.append(f"- Safety: {safety}")
    if partner != "—":  lines.append(f"- Partner required: {partner}")
    if solo != "—":     lines.append(f"- Solo: {solo}")
    lines.append(f"- Definition: {desc if desc else '(not listed).'}")
    return "\n".join(lines)

# ------------------------- CSV helpers -------------------------

def _iter_csv_like_lines(md_text: str):
    """Yield non-empty, non-heading lines that contain commas."""
    for raw in (md_text or "").splitlines():
        st = raw.strip()
        if not st:
            continue
        if st.startswith("#") or st.startswith("```"):
            continue
        if "," in raw:
            yield raw

def _split_row_limited(raw: str) -> List[str]:
    """
    Split a CSV-like row into EXPECTED_COLS pieces.
    The last field absorbs the remaining commas (so descriptions with commas are safe).
    """
    parts = raw.split(",", EXPECTED_COLS - 1)
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
    return {
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

# ---------------------- NEW: direct line lookup ----------------------

def _direct_line_lookup(md_text: str, cand_variants: List[str]) -> Optional[Dict[str, Any]]:
    """
    Exactly match the first cell (technique name) of a CSV line against
    name variants (case/macrón-insensitive). Fast and robust for your
    single-line-per-technique markdown.
    """
    anchors = {_fold(v) for v in cand_variants if v and not v.startswith("#")}
    if not anchors:
        return None

    for raw in (md_text or "").splitlines():
        line = raw.rstrip()
        if not line or "," not in line:
            continue
        first = line.split(",", 1)[0].strip()
        if _fold(first) in anchors:
            row = _split_row_limited(line)
            return _row_to_record_positional(row)
    return None

# ------------------------- CSV table fallback -------------------------

def _csv_fallback_lookup(md_text: str, cand_variants: List[str]) -> Optional[Dict[str, Any]]:
    rows = _scan_csv_rows_limited(md_text)
    if not rows:
        return None

    has_header = _has_header(rows[0])
    data_rows = rows[1:] if has_header else rows

    cand_folded = [_fold(c) for c in cand_variants]
    cand_lite = [_lite(c) for c in cand_variants]

    # 1) exact-ish key hits
    for r in data_rows:
        if not r or not r[0].strip():
            continue
        name = r[0].strip()
        name_fold = _fold(name)
        name_lite = _lite(name)
        if name_fold in cand_folded or name_lite in cand_lite:
            return _row_to_record_positional(r)

    # 2) fuzzy best match (high threshold to avoid wrong hits)
    best = (None, 0.0)
    target = _fold(cand_variants[0]) if cand_variants else ""
    for r in data_rows:
        name = (r[0] or "").strip()
        if not name:
            continue
        s = SequenceMatcher(None, _fold(name), target).ratio()
        if s > best[1]:
            best = (r, s)

    if best[0] is not None and best[1] >= 0.85:
        return _row_to_record_positional(best[0])
    return None

# ---------------------- public entrypoint ----------------------

def try_answer_technique(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """Return a structured technique answer or None to fall back upstream."""
    if not _looks_like_technique_q(question):
        return None

    md_text = _gather_full_technique_text(passages)
    if not md_text.strip():
        return None

    ql = _norm_space(question).lower()
    cand_raw = _extract_candidate(ql)
    variants = _candidate_variants(cand_raw)

    # 1) Direct line lookup (most reliable for your CSV-like .md format)
    rec = _direct_line_lookup(md_text, variants)
    if rec:
        return _format_bullets(rec)

    # 2) CSV table lookup
    rec = _csv_fallback_lookup(md_text, variants)
    if rec:
        return _format_bullets(rec)

    # 3) Optional indexed route (kept for compatibility)
    if parse_technique_md and build_indexes:
        try:
            records = parse_technique_md(md_text)
            idx = build_indexes(records) if records else None
        except Exception:
            idx = None
        if idx:
            by_name = idx["by_name"]; by_lower = idx["by_lower"]
            by_fold = idx["by_fold"]; by_key = idx["by_keylite"]

            # Direct dictionary hits
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

            # Fuzzy across names
            cq = _fold(cand_raw)
            best_name, best_score = None, 0.0
            for name in by_name.keys():
                s = SequenceMatcher(None, cq, _fold(name)).ratio()
                if s > best_score:
                    best_name, best_score = name, s
            if best_name and best_score >= 0.80:
                return _format_bullets(by_name[best_name])

    return None
