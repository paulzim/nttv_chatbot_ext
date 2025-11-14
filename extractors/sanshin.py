# extractors/sanshin.py
from __future__ import annotations
import re
from typing import List, Dict, Any, Optional
from .common import join_oxford, dedupe_preserve

# ------------------------------------------------------------
# Data: canonical Sanshin elements
# ------------------------------------------------------------

_ELEMENT_DATA: Dict[str, Dict[str, Any]] = {
    "chi": {
        "name": "Chi no Kata",
        "english": "Earth Form",
        "aliases": ["chi no kata", "earth form"],
        "summary": (
            "Chi no Kata (Earth Form) emphasizes grounding, structure, and "
            "a strong, stable base. Movements tend to sink and rise, teaching "
            "you to connect to the ground and generate power from the legs and hips."
        ),
    },
    "sui": {
        "name": "Sui no Kata",
        "english": "Water Form",
        "aliases": ["sui no kata", "water form"],
        "summary": (
            "Sui no Kata (Water Form) focuses on flowing, outward-inward and circular "
            "movement. It trains adaptability, continuous motion, and the ability to "
            "redirect force rather than meeting it head-on."
        ),
    },
    "ka": {
        "name": "Ka no Kata",
        "english": "Fire Form",
        "aliases": ["ka no kata", "fire form"],
        "summary": (
            "Ka no Kata (Fire Form) develops sharp, accelerating strikes with a twisting "
            "quality. It represents expansion, intensity, and the ability to explode "
            "through an opponent's guard."
        ),
    },
    "fu": {
        "name": "Fu no Kata",
        "english": "Wind Form",
        "aliases": ["fu no kata", "wind form"],
        "summary": (
            "Fu no Kata (Wind Form) trains light, off-line movement and angled entries. "
            "It emphasizes evasion, changing position, and striking from unexpected "
            "angles like the movement of wind around obstacles."
        ),
    },
    "ku": {
        "name": "Ku no Kata",
        "english": "Void Form",
        "aliases": ["ku no kata", "void form"],
        "summary": (
            "Ku no Kata (Void Form) expresses timing, distance, and the use of space. "
            "It represents emptiness and potential, teaching you to move at the right "
            "moment and appear where the opponent is unprepared."
        ),
    },
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def _looks_like_sanshin_question(question: str) -> bool:
    q = _norm(question)
    if "sanshin" in q or "san shin" in q:
        return True
    # element-specific without the word 'sanshin'
    for meta in _ELEMENT_DATA.values():
        for alias in meta["aliases"]:
            if alias in q:
                return True
    return False

def _detect_element(question: str) -> Optional[Dict[str, Any]]:
    q = _norm(question)
    for meta in _ELEMENT_DATA.values():
        for alias in meta["aliases"]:
            if alias in q:
                return meta
    return None

def _wants_list(question: str) -> bool:
    q = _norm(question)
    return (
        ("what are" in q or "list" in q or "which" in q or "name the" in q)
        and ("sanshin" in q or "san shin" in q or "five elements" in q or "5 elements" in q)
    )

def _wants_overview(question: str) -> bool:
    q = _norm(question)
    if "what is" in q or "explain" in q or "describe" in q:
        if "sanshin" in q or "san shin" in q:
            return True
    # fallback if they literally type "sanshin no kata"
    if "sanshin no kata" in q or "san shin no kata" in q:
        return True
    return False

# Legacy-style helpers kept so existing imports don't break,
# even if we don't rely on them heavily now.
def _collect_after_anchor(blob: str, anchor_regex: str, window: int = 3000) -> str:
    m = re.search(anchor_regex, blob, flags=re.I)
    if not m:
        return ""
    return blob[m.end() : m.end() + window]

def _parse_bullets_or_shortlines(seg: str) -> List[str]:
    lines, started = [], False
    for raw in seg.splitlines():
        s = raw.strip()
        if not s:
            if started:
                break
            continue
        if s.startswith(("·", "-", "*", "•")):
            started = True
            lines.append(s.lstrip("·-*• ").strip())
        elif started:
            # Stop if we hit a non-bullet after we've started
            break
    return lines

# ------------------------------------------------------------
# Public entrypoint
# ------------------------------------------------------------

def try_answer_sanshin(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic Sanshin no Kata extractor.

    Handles:
      * 'what is Sanshin no Kata?'
      * 'explain the Sanshin'
      * 'what are the five elements of Sanshin no Kata?'
      * 'what is Chi no Kata?'
      * 'describe Sui no Kata', etc.
    """
    if not _looks_like_sanshin_question(question):
        return None

    # Element-specific questions
    elem = _detect_element(question)
    if elem is not None:
        name = elem["name"]
        eng = elem["english"]
        summary = elem["summary"]
        return f"{name} ({eng}): {summary}"

    # List-style questions about the elements
    if _wants_list(question):
        ordered = [meta["name"] for meta in _ELEMENT_DATA.values()]
        ordered = dedupe_preserve(ordered)
        if len(ordered) >= 3:
            return (
                "Sanshin no Kata (Five Elements) consists of "
                + join_oxford(ordered)
                + "."
            )

    # Overview of Sanshin no Kata
    if _wants_overview(question):
        names = [meta["name"] for meta in _ELEMENT_DATA.values()]
        names = dedupe_preserve(names)
        elements_list = join_oxford(names)
        return (
            "Sanshin no Kata (Three Hearts / Five Elements) is a set of five fundamental "
            "solo forms used in the Bujinkan to train body structure, timing, and feeling. "
            "Each form is associated with an element and a characteristic way of moving. "
            f"The five Sanshin forms are {elements_list}."
        )

    # Fallback: if they typed something like 'Sanshin?' with no other cue
    names = [meta["name"] for meta in _ELEMENT_DATA.values()]
    elements_list = join_oxford(dedupe_preserve(names))
    return (
        "Sanshin no Kata consists of "
        + elements_list
        + "."
    )
