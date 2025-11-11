# extractors/techniques.py
from __future__ import annotations
import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional

# These imports are safe even if the helper returns partial fields;
# we won’t trust description from here unless it exists.
try:
    from .technique_loader import parse_technique_md, build_indexes
except Exception:
    # Make them optional; CSV fallback will still work.
    parse_technique_md = None
    build_indexes = None

# Concept words we never treat as specific techniques
CONCEPT_BANS = ("kihon happo", "sanshin", "school", "schools", "ryu", "ryū")

TRIGGERS = ("what is", "define", "explain")
NAME_HINTS = (
    "gyaku", "dori", "kudaki", "gatame", "otoshi", "nage", "seoi", "kote",
    "musha", "take ori", "juji", "omote", "ura", "ganseki", "hodoki",
    "kata", "no kata",
)

# Expected positional schema for fallback rows (12 columns)
#  0 name
#  1 japanese
#  2 translation
#  3 type
#  4 rank
#  5 in_rank
#  6 primary_focus
#  7 safety
#  8 partner_required
#  9 solo
# 10 tags
# 11 description  <-- absorbs ALL remaining commas
EXPECTED_COLS = 12

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
    # hyphen insensitivity
    raw_no_hy = raw.replace("-", " ")
    if raw_no_hy != raw:
        v.append(raw_no_hy)
        if raw_no_hy.lower().endswith(" no kata"):
            v.append(raw_no_hy[:-8].strip())
        else:
            v.append(f"{raw_no_hy} no kata")
    # folded + lite forms
    v.append(_fold(raw))
    v.append(_lite(raw))
    # de-dupe preserve order
    seen = set(); out = []
    for x in v:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def _wordhit(alias: str, hay: str) -> bool:
    if not alias:
        return False
    pat = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
    return re.search(pat, hay) is not None

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

# ---------- CSV fallback (manual limited split; description absorbs commas) ----------

def _iter_csv_like_lines(md_text: str):
    """Yield CSV-like lines from markdown, skipping headings/fences/empties."""
    for raw in (md_text or "").splitlines():
        st = raw.strip()
        if not st:
            continue
        if st.startswith("#") or st.startswith("```"):
            continue
        # only treat lines that look like rows (comma-separated)
        if "," in raw:
            yield raw

def _split_row_limited(raw: str) -> List[str]:
    """
    Split a CSV-like line by commas, but only the first 11 commas become columns.
    Column 12 (index 11) = 'description' absorbs the remainder (commas allowed).
    """
    parts = raw.split(",", EXPECTED_COLS - 1)
    # strip whitespace around each cell
    parts = [p.strip() for p in parts]
    # If there are MORE than EXPECTED_COLS (shouldn’t happen with maxsplit),
    # fold extras into the last slot for safety.
    if len(parts) > EXPECTED_COLS:
        head = parts[:EXPECTED_COLS-1]
        tail = ",".join(parts[EXPECTED_COLS-1:])
        parts = head + [tail]
    # pad missing cells
    if len(parts) < EXPECTED_COLS:
        parts += [""] * (EXPECTED_COLS - len(parts))
    return parts

def _scan_csv_rows_limited(md_text: str) -> List[List[str]]:
    return [_split_row_limited(raw) for raw in _iter_csv_like_lines(md_text)]

def _has_header(cells: List[str]) -> bool:
    header = [c.strip().lower() for c in cells]
    return any(h in {"name","translation","japanese","description"} for h in header)

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
        "partner_required": row[8],
        "solo": row[9],
        "tags": row[10],
        "description": row[11],
    }
    # best-effort booleans
    def to_bool(x: str) -> Optional[bool]:
        v = (x or "").strip().lower()
        if v in {"1","true","yes","y","✅","✓","✔"}: return True
        if v in {"0","false","no","n","❌","✗","✕"}: return False
        return None
    rec["partner_required"] = to_bool(rec.get("partner_required") or "")
    rec["solo"] = to_bool(rec.get("solo") or "")
    return rec

def _csv_fallback_lookup(md_text: str, cand_variants: List[str]) -> Optional[Dict[str, Any]]:
    rows = _scan_csv_rows_limited(md_text)
    if not rows:
        return None

    # Detect header
    has_header = _has_header(rows[0])
    data_rows = rows[1:] if has_header else rows

    # Folded variants
    cand_folded = [_fold(c) for c in cand_variants]
    cand_lite = [_lite(c) for c in cand_variants]

    best = (None, 0.0, None)  # (row, score, name)
    for r in data_rows:
        if not r or not r[0].strip():
            continue
        name = r[0].strip()
        name_fold = _fold(name)
        name_lite = _lite(name)

        # exact-ish
        if name_fold in cand_folded or name_lite in cand_lite:
            return _row_to_record_positional(r)

        # fuzzy
        score = max(SequenceMatcher(None, name_fold, cf).ratio() for cf in cand_folded)
        if score > best[1]:
            best = (r, score, name)

    if best[0] is not None and best[1] >= 0.85:
        return _row_to_record_positional(best[0])

    return None

# ---------- main entry ----------

def try_answer_technique(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic lookup from Technique Descriptions.md.
    NEW: Prefer CSV fallback (robust to commas in description). If that fails, try the
    indexed path; if the indexed record lacks a description, retry CSV before returning.
    """
    if not _looks_like_technique_q(question):
        return None

    md_text = _gather_full_technique_text(passages)
    if not md_text.strip():
        return None

    ql = _norm_space(question).lower()
    cand_raw = _extract_candidate(ql)
    variants = _candidate_variants(cand_raw)

    # 1) CSV fallback FIRST (more robust for description field)
    rec = _csv_fallback_lookup(md_text, variants)
    if rec:
        return _format_bullets(rec)

    # 2) Indexed route (only if helper exists)
    if parse_technique_md and build_indexes:
        try:
            records = parse_technique_md(md_text)
            idx = build_indexes(records) if records else None
        except Exception:
            idx = None
        if idx:
            by_name = idx["by_name"]; by_lower = idx["by_lower"]
            by_fold = idx["by_fold"]; by_key = idx["by_keylite"]

            # direct key lookups
            for v in variants:
                key = v.lower()
                if key in by_lower:
                    rec2 = by_name[by_lower[key]]
                    # If indexed rec is missing description, retry CSV:
                    if not (rec2.get("description") or "").strip():
                        rec = _csv_fallback_lookup(md_text, variants)
                        if rec:
                            return _format_bullets(rec)
                    return _format_bullets(rec2)

                fkey = _fold(v)
                if fkey in by_fold:
                    rec2 = by_name[by_fold[fkey]]
                    if not (rec2.get("description") or "").strip():
                        rec = _csv_fallback_lookup(md_text, variants)
                        if rec:
                            return _format_bullets(rec)
                    return _format_bullets(rec2)

                lkey = _lite(v)
                if lkey in by_key:
                    rec2 = by_name[by_key[lkey]]
                    if not (rec2.get("description") or "").strip():
                        rec = _csv_fallback_lookup(md_text, variants)
                        if rec:
                            return _format_bullets(rec)
                    return _format_bullets(rec2)

            # whole-word checks
            for table in (by_lower, by_fold):
                for alias, canon in table.items():
                    if _wordhit(alias, cand_raw) or _wordhit(alias, ql):
                        rec2 = by_name[canon]
                        if not (rec2.get("description") or "").strip():
                            rec = _csv_fallback_lookup(md_text, variants)
                            if rec:
                                return _format_bullets(rec)
                        return _format_bullets(rec2)

            # fuzzy to canonical names
            cq = _fold(cand_raw)
            best_name, best_score = None, 0.0
            for name in by_name.keys():
                s = SequenceMatcher(None, cq, _fold(name)).ratio()
                if s > best_score:
                    best_name, best_score = name, s
            if best_name and best_score >= 0.80:
                rec2 = by_name[best_name]
                if not (rec2.get("description") or "").strip():
                    rec = _csv_fallback_lookup(md_text, variants)
                    if rec:
                        return _format_bullets(rec)
                return _format_bullets(rec2)

    # 3) No match
    return None
