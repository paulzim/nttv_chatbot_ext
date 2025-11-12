# extractors/kihon_happo.py
import re
from typing import List, Dict, Any, Optional

CANON_DEF = "Kihon Happo consists of Kosshi Kihon Sanpo and Torite Goho."

UNWANTED_HINTS = (
    "drill the kihon happo",
    "practice the kihon happo",
    "use it against attackers",
    "from all kamae",
)

def _scrub_training_line(s: str) -> bool:
    ls = s.lower()
    return any(h in ls for h in UNWANTED_HINTS)

def _extract_lists(text: str) -> (List[str], List[str]):
    """Pull Kosshi Kihon Sanpo and Torite Goho lists from a block of text."""
    kosshi, torite = [], []

    for raw in (text or "").splitlines():
        ln = raw.strip()
        if not ln:
            continue
        low = ln.lower()

        # Skip training/drill lines entirely
        if _scrub_training_line(ln):
            continue

        # Kosshi list
        if "kosshi" in low and "sanpo" in low:
            tail = ln.split(":", 1)[1].strip() if ":" in ln else ln
            parts = [p.strip(" -•\t") for p in re.split(r"[;,]", tail)]
            kosshi.extend([p for p in parts if 2 <= len(p) <= 60])

        # Torite list (accept goho/gohō)
        if "torite" in low and ("goho" in low or "gohō" in low):
            tail = ln.split(":", 1)[1].strip() if ":" in ln else ln
            parts = [p.strip(" -•\t") for p in re.split(r"[;,]", tail)]
            torite.extend([p for p in parts if 2 <= len(p) <= 60])

    # Dedup, keep order, cap
    def dedupe(seq: List[str]) -> List[str]:
        seen = set(); out: List[str] = []
        for x in seq:
            if x and x not in seen:
                out.append(x); seen.add(x)
        return out

    kosshi = dedupe(kosshi)[:3]
    torite  = dedupe(torite)[:5]
    return kosshi, torite

def try_answer_kihon_happo(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    ql = (question or "").lower()
    if "kihon happo" not in ql and "kihon happō" not in ql:
        return None

    kosshi, torite = [], []

    # Scan retrieved passages (you already inject a synthetic Kihon block upstream)
    for p in passages[:12]:
        k, t = _extract_lists(p.get("text", ""))
        if k: kosshi = k
        if t: torite = t
        if kosshi and torite:
            break

    # If lists are missing, return the canonical one-liner instead of a noisy training line
    if not kosshi and not torite:
        return CANON_DEF

    # Deterministic output; always lead with the canonical definition
    lines = ["Kihon Happo:"]
    lines.append(f"- {CANON_DEF}")
    if kosshi:
        lines.append(f"- Kosshi Kihon Sanpo: {', '.join(kosshi)}.")
    if torite:
        lines.append(f"- Torite Goho: {', '.join(torite)}.")
    return "\n".join(lines)
