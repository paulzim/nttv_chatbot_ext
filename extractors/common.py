# extractors/common.py
import re
from typing import Iterable, List

def join_oxford(items: Iterable[str]) -> str:
    items = [x.strip() for x in items if x and x.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]

def dedupe_preserve(seq: Iterable[str]) -> List[str]:
    out, seen = [], set()
    for x in seq:
        lx = x.strip().lower()
        if lx not in seen:
            out.append(x.strip())
            seen.add(lx)
    return out

BULLET_RE = re.compile(r"^[-·•]\s+")
TITLELIKE_RE = re.compile(r'^[A-Z][A-Za-z0-9\s"’\-\(\)]+$')
