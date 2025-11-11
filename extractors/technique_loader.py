# extractors/technique_loader.py
from __future__ import annotations
import csv
import io
import re
import unicodedata
from typing import Dict, List, Any, Optional

FIELDS_CANON = [
    "name", "japanese", "translation", "type", "rank",
    "in_rank", "primary_focus", "safety",
    "partner_required", "solo",
    "tags", "description",
]

BOOL_TRUE = {"1", "true", "yes", "y", "✅", "✓", "✔"}
BOOL_FALSE = {"0", "false", "no", "n", "❌", "✗", "✕"}

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _fold(s: str) -> str:
    # Unicode fold: strip macrons/accents and lowercase
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()

def _keylite(s: str) -> str:
    # Aggressive key: folded, alnum-only
    s = _fold(s)
    return re.sub(r"[^a-z0-9]+", "", s)

def _to_bool(s: str) -> Optional[bool]:
    if s is None:
        return None
    v = _norm(s).lower()
    if v in BOOL_TRUE:
        return True
    if v in BOOL_FALSE:
        return False
    return None

def _split_tags(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[|,]", s)
    return [t.strip() for t in parts if t.strip()]

def _canon_header(h: str) -> str:
    h = _norm(h).lower()
    maps = {
        "japanese name": "japanese",
        "jp": "japanese",
        "name jp": "japanese",
        "trans": "translation",
        "focus": "primary_focus",
        "rank_intro": "rank",
        "rank-intro": "rank",
        "rankintro": "rank",
        "in rank": "in_rank",
        "partner": "partner_required",
        "tag": "tags",
    }
    return maps.get(h, h)

def parse_technique_md(md_text: str) -> List[Dict[str, Any]]:
    """
    Parse CSV rows contained in a Markdown file.
    Skips headings/code fences; accepts headered or headerless CSV.
    """
    lines = []
    for raw in (md_text or "").splitlines():
        st = raw.strip()
        if not st or st.startswith("#") or st.startswith("```"):
            continue
        if "," in raw:
            lines.append(raw)

    if not lines:
        return []

    reader = csv.reader(io.StringIO("\n".join(lines)))
    rows = list(reader)
    if not rows:
        return []

    header = [_canon_header(h) for h in rows[0]]
    header_is_valid = any(h in FIELDS_CANON for h in header)
    data_rows = rows[1:] if header_is_valid else rows

    if not header_is_valid:
        header = FIELDS_CANON[:]

    out: List[Dict[str, Any]] = []
    for r in data_rows:
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
        elif len(r) > len(header):
            r = r[:len(header)]

        rec = { header[i]: (r[i].strip() if i < len(r) else "") for i in range(len(header)) }

        # normalize
        rec["name"] = _norm(rec.get("name", ""))
        if not rec["name"]:
            continue

        rec["japanese"] = _norm(rec.get("japanese", ""))
        rec["translation"] = _norm(rec.get("translation", ""))
        rec["type"] = _norm(rec.get("type", ""))
        rec["rank"] = _norm(rec.get("rank", ""))
        rec["primary_focus"] = _norm(rec.get("primary_focus", ""))
        rec["safety"] = _norm(rec.get("safety", ""))
        rec["in_rank"] = _to_bool(rec.get("in_rank", ""))
        rec["partner_required"] = _to_bool(rec.get("partner_required", ""))
        rec["solo"] = _to_bool(rec.get("solo", ""))
        rec["tags"] = _split_tags(rec.get("tags", ""))
        rec["description"] = (rec.get("description", "") or "").strip()

        out.append(rec)

    return out

def build_indexes(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build multiple lookups for robust matching.
    - by_name: canonical name -> rec
    - by_lower / by_fold / by_keylite: alias -> canonical name
    """
    by_name: Dict[str, Dict[str, Any]] = {}
    by_lower: Dict[str, str] = {}
    by_fold: Dict[str, str] = {}
    by_keylite: Dict[str, str] = {}

    def add_alias(alias: str, canon: str):
        if not alias:
            return
        low = alias.lower()
        fol = _fold(alias)
        key = _keylite(alias)
        by_lower[low] = canon
        by_fold[fol] = canon
        by_keylite[key] = canon

    for r in records:
        name = r["name"]
        by_name[name] = r

        # name
        add_alias(name, name)

        # add "no kata" synthetics (e.g., "Jumonji" -> "Jumonji no Kata")
        if name.lower().endswith(" no kata"):
            root = name[:-8].strip()
            add_alias(root, name)
        else:
            add_alias(f"{name} no kata", name)

        # translation & japanese
        add_alias(r.get("translation", ""), name)
        add_alias(r.get("japanese", ""), name)

        # tags
        for t in r.get("tags", []):
            add_alias(t, name)

    return {
        "by_name": by_name,
        "by_lower": by_lower,
        "by_fold": by_fold,
        "by_keylite": by_keylite,
    }
