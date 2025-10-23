# extractors/kihon_happo.py
import re
from typing import List, Dict, Any
from .common import dedupe_preserve

def try_answer_kihon_happo(passages: List[Dict[str, Any]]) -> str | None:
    """
    Deterministic Kihon Happo answer — keeps Kosshi and Torite cleanly separated.
    If Torite bullets aren't captured under the anchor, fall back to a global scan.
    """
    blob = "\n\n".join(p["text"] for p in passages[:6])
    blob_low = blob.lower()
    if "kihon happo" not in blob_low:
        return None

    # --- anchors / segments (flexible about spacing/case)
    kosshi_match = re.search(
        r"(?is)kosshi\s+kihon\s+sanpo\s*:?\s*(.*?)(?:\n\s*(?:torite\s+goho\s+gata)\b|$)",
        blob
    )
    torite_match = re.search(
        r"(?is)torite\s+goho\s+gata\s*:?\s*(.*?)(?:\n\s*\n|$)",
        blob
    )

    def _parse_bullets(seg: str | None) -> List[str]:
        if not seg:
            return []
        out, started = [], False
        for raw in seg.splitlines():
            r = raw.strip()
            if not r:
                if started:
                    break
                continue
            if re.match(r"^[-·•]\s+", r):
                started = True
                r = re.sub(r"^[-·•]\s+", "", r)
                out.append(r)
            else:
                # allow a few short title-like lines before bullets start
                if not started and re.match(r'^[A-Z][A-Za-z0-9\s"’\-–—]+$', r) and len(r.split()) <= 7:
                    out.append(r)
                elif started:
                    break
        return dedupe_preserve(out)

    kosshi_raw = _parse_bullets(kosshi_match.group(1) if kosshi_match else None)
    torite_raw = _parse_bullets(torite_match.group(1) if torite_match else None)

    # --- Kosshi: only kata names
    kata_terms = ("kata", "ichimonji", "hicho", "jumonji")
    kosshi_items = [x for x in kosshi_raw if any(t in x.lower() for t in kata_terms)]

    # --- Torite: canonical order / robust scan
    torite_targets = [
        (r"\bomote\s+gyaku\b", "Omote Gyaku"),
        (r"\bomote\s+gyaku\s+ken\s+sabaki\b|\bken\s+sabaki\b", "Omote Gyaku Ken Sabaki"),
        (r"\bura\s+gyaku\b", "Ura Gyaku"),
        (r"\bmusha\s+dori\b", "Musha Dori"),
        (r"\bganseki\s+nage\b", "Ganseki Nage"),
    ]

    torite_block_terms = ("kata", "ichimonji", "hicho", "jumonji")
    torite_keep_terms  = ("gyaku", "dori", "nage", "sabaki")
    torite_items = [
        x for x in torite_raw
        if any(t in x.lower() for t in torite_keep_terms)
        and not any(b in x.lower() for b in torite_block_terms)
    ]

    # Fallback: scan entire blob in canonical order
    if not torite_items:
        found = []
        for pat, canon in torite_targets:
            if re.search(pat, blob, flags=re.I):
                found.append(canon)
        torite_items = found

    # Final de-dupe and separation
    kosshi_set = {x.lower() for x in kosshi_items}
    seen = set()
    torite_items = [
        x for x in torite_items
        if x.lower() not in seen and x.lower() not in kosshi_set and not seen.add(x.lower())
    ]

    if not kosshi_items and not torite_items:
        return None

    left  = "Kosshi Kihon Sanpo" + (f" ({', '.join(kosshi_items)})" if kosshi_items else "")
    right = "Torite Goho Gata"   + (f" ({', '.join(torite_items)})"  if torite_items else "")

    return f"Kihon Happo consists of {left} and {right}."
