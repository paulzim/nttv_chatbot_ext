# extractors/kihon_happo.py
import re
from typing import List, Dict, Any, Optional

WANTED_DEF_STARTS = (
    "kihon happo consists of",
    "kihon happō consists of",
)
UNWANTED_DEF_HINTS = (
    "drill the kihon happo",
    "practice the kihon happo",
    "use it against attackers",
)

def try_answer_kihon_happo(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    ql = question.lower()
    if "kihon happo" not in ql and "kihon happō" not in ql:
        return None

    kosshi, torite, defs = [], [], []

    def push(text: str):
        for raw in (text or "").splitlines():
            ln = raw.strip()
            if not ln:
                continue
            low = ln.lower()

            # Definition lines: prefer canonical phrasing and avoid drill/training lines
            if "kihon happo" in low:
                if any(low.startswith(x) for x in WANTED_DEF_STARTS) and not any(u in low for u in UNWANTED_DEF_HINTS):
                    defs.append(ln.rstrip(" ;,"))
                # fallback: short neutral mention
                elif ("consists of" in low or "made up of" in low) and not any(u in low for u in UNWANTED_DEF_HINTS):
                    defs.append(ln.rstrip(" ;,"))

            # Kosshi Kihon Sanpo items
            if "kosshi" in low and "sanpo" in low:
                tail = ln.split(":", 1)[1].strip() if ":" in ln else ln
                parts = [p.strip(" -•\t") for p in re.split(r"[;,]", tail) if 2 <= len(p.strip()) <= 60]
                kosshi.extend(parts)

            # Torite Goho items
            if "torite" in low and ("goho" in low or "gohō" in low):
                tail = ln.split(":", 1)[1].strip() if ":" in ln else ln
                parts = [p.strip(" -•\t") for p in re.split(r"[;,]", tail) if 2 <= len(p.strip()) <= 60]
                torite.extend(parts)

    # scan retrieved first
    for p in passages[:8]:
        push(p.get("text", ""))

    # fallback: light scan of CHUNKS if needed (app will prepend synthetic already)
    if (len(kosshi) < 3 or len(torite) < 5) and defs:
        kosshi = kosshi[:3]
        torite = torite[:5]
    else:
        kosshi = kosshi[:3]
        torite = torite[:5]

    if not (kosshi or torite or defs):
        return None

    parts = ["Kihon Happo consists of Kosshi Kihon Sanpo and Torite Goho."]
    if kosshi:
        parts.append("Kosshi Kihon Sanpo: " + ", ".join(kosshi) + ".")
    if torite:
        parts.append("Torite Goho: " + ", ".join(torite) + ".")
    if defs:
        # Prefer the first good canonical sentence; append only if not redundant
        if not parts[-1].endswith("."):
            parts[-1] += "."
        parts.append(defs[0])

    return "\n".join(parts)
