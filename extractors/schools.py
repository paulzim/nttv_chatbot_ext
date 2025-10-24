# extractors/schools.py
import re
from typing import List, Dict, Any

def try_answer_schools(passages: List[Dict[str, Any]]) -> str | None:
    """
    Deterministic extractor for 'schools of the Bujinkan'.
    Searches for the nine school names anywhere in the retrieved text.
    """
    blob = "\n\n".join(p["text"] for p in passages[:8])
    blob_low = blob.lower()
    if "bujinkan" not in blob_low:
        return None

    target_schools = [
        "togakure ryu",
        "gyokko ryu",
        "koto ryu",
        "shinden fudo ryu",
        "kukishinden ryu",
        "takagi yoshin ryu",
        "gyokushin ryu",
        "kumogakure ryu",
        "gikan ryu",
    ]

    found = []
    for s in target_schools:
        if s in blob_low:
            m = re.search(rf"({s})", blob, flags=re.I)
            found.append(m.group(1) if m else s.title())

    if found:
        return "The nine schools of the Bujinkan are: " + ", ".join(found) + "."
    return None
