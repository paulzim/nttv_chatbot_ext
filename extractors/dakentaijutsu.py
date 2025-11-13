# extractors/dakentaijutsu.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import unicodedata
import re

from .common import join_oxford


# ----------------- small helpers -----------------


def _fold(s: str) -> str:
    """Case- and accent-insensitive fold."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _looks_like_daken_question(question: str) -> bool:
    """
    Only fire this extractor when the user clearly means Dakentaijutsu /
    striking content, so we don't fight with the technique CSV.

    Triggers:
      - 'dakentaijutsu'
      - 'hoken juroppo' / 'sixteen hidden fists' / 'sixteen secret fists'
      - 'uke nagashi'
      - 'principles of striking'
      - generic 'kicks' / 'blocks' / 'striking' questions without a rank
    """
    q = _fold(question)

    if "dakentaijutsu" in q or "daken taijutsu" in q:
        return True

    if "hoken juroppo" in q:
        return True
    if "sixteen hidden fists" in q or "sixteen secret fists" in q or "sixteen fists" in q:
        return True

    if "uke nagashi" in q:
        return True

    if "principles of striking" in q:
        return True

    # Generic list-style kick / block questions
    if ("kick" in q or "kicks" in q or "geri" in q) and "taihenjutsu" not in q:
        return True

    if ("block" in q or "blocks" in q or "blocking" in q) and "taihenjutsu" not in q:
        return True

    return False


# ----------------- file loading -----------------


def _data_dir() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent / "data"


def _load_training_text() -> str:
    p = _data_dir() / "nttv training reference.txt"
    try:
        if p.exists():
            return p.read_text(encoding="utf-8")
    except Exception:
        return ""
    return ""


# ----------------- parse Dakentaijutsu block -----------------


def _extract_daken_block() -> str:
    """
    Extract the Dakentaijutsu section:

        Dakentaijutsu- Striking and Blocking Skills
        ...
        Dakentaijutsu- Striking Techniques (cont.)
        ...
        Keri- Kicks
        ...
        Uke Nagashi- Receiving Flow
        ...

    We stop at 'Kihon Happo- The Eight Basic Ways'.
    """
    text = _load_training_text()
    if not text:
        return ""

    start = text.find("Dakentaijutsu- Striking and Blocking Skills")
    if start == -1:
        return ""

    end_marker = "Kihon Happo- The Eight Basic Ways"
    end = text.find(end_marker, start)
    if end == -1:
        end = None

    return text[start:end]


def _parse_daken_records() -> (Dict[str, Dict[str, str]], str):
    """
    Parse the Dakentaijutsu block into:

        records: name -> {name, desc, category}
        hoken_desc: description text from the Hoken Juroppo Ken header
    """
    block = _extract_daken_block()
    if not block:
        return {}, ""

    records: Dict[str, Dict[str, str]] = {}
    current_cat: Optional[str] = None
    hoken_desc = ""

    valid_cats = {
        "Blocking",
        "Striking",
        "Hoken Juroppo Ken",
        "Kicks",
        "Uke Nagashi",
        "Principles",
    }

    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Section headings
        if line.startswith("Blocking"):
            current_cat = "Blocking"
            continue

        if line.startswith("Striking"):
            current_cat = "Striking"
            continue

        if line.startswith("Dakentaijutsu- Striking Techniques"):
            # continuation of striking section
            current_cat = "Striking"
            continue

        if line.startswith("Hoken Juroppo Ken-"):
            current_cat = "Hoken Juroppo Ken"
            # e.g. "Hoken Juroppo Ken- The Sixteen Hidden/Secret Fists"
            parts = line.split("-", 1)
            if len(parts) == 2:
                hoken_desc = parts[1].strip()
            continue

        if line.startswith("Principles of Striking"):
            current_cat = "Principles"
            continue

        if line.startswith("Keri-"):
            current_cat = "Kicks"
            continue

        if line.startswith("Uke Nagashi-"):
            current_cat = "Uke Nagashi"
            continue

        # Non-Dakentaijutsu headings we want to skip (NOTES, Zanshin etc.)
        if line.startswith("NOTES") or line.startswith("Zanshin-"):
            current_cat = None
            continue

        if current_cat not in valid_cats:
            continue

        # Bullet lines
        if line.startswith(("·", "-", "*", "•")):
            body = line[1:].strip()
            if not body:
                continue

            if "-" in body:
                name_part, desc_part = body.split("-", 1)
                name = name_part.strip()
                desc = desc_part.strip()
            else:
                name = body.strip()
                desc = ""

            if not name:
                continue

            key = _fold(name)
            records[key] = {
                "name": name,
                "desc": desc,
                "category": current_cat,
            }

    return records, hoken_desc


# ----------------- answering helpers -----------------


def _answer_hoken_list(hoken_desc: str, records: Dict[str, Dict[str, str]]) -> Optional[str]:
    """
    Answer questions about Hoken Juroppo Ken (Sixteen Hidden/Secret Fists).
    """
    fists = [r["name"] for r in records.values() if r.get("category") == "Hoken Juroppo Ken"]
    if not fists:
        return None

    title = "Hoken Juroppo Ken"
    if hoken_desc:
        header = f"{title} ({hoken_desc})"
    else:
        header = title

    return f"{header}: {join_oxford(fists)}"


def _answer_kicks_list(records: Dict[str, Dict[str, str]]) -> Optional[str]:
    kicks = [r["name"] for r in records.values() if r.get("category") == "Kicks"]
    if not kicks:
        return None
    return f"Dakentaijutsu kicks (Keri): {join_oxford(kicks)}"


def _answer_blocks_list(records: Dict[str, Dict[str, str]]) -> Optional[str]:
    blocks = [r["name"] for r in records.values() if r.get("category") == "Blocking"]
    if not blocks:
        return None
    return f"Basic Dakentaijutsu blocks: {join_oxford(blocks)}"


def _answer_uke_nagashi(records: Dict[str, Dict[str, str]]) -> Optional[str]:
    subs = [r["name"] for r in records.values() if r.get("category") == "Uke Nagashi"]
    if not subs:
        return None
    return f"Uke Nagashi (Receiving Flow) variations: {join_oxford(subs)}"


def _answer_principles(records: Dict[str, Dict[str, str]]) -> Optional[str]:
    items = [r["name"] for r in records.values() if r.get("category") == "Principles"]
    if not items:
        return None
    return "Principles of striking: " + "; ".join(items)


def _answer_list_style(question: str, records: Dict[str, Dict[str, str]], hoken_desc: str) -> Optional[str]:
    q = _fold(question)

    # Hoken Juroppo Ken
    if "hoken juroppo" in q or "sixteen hidden fists" in q or "sixteen secret fists" in q or "sixteen fists" in q:
        ans = _answer_hoken_list(hoken_desc, records)
        if ans:
            return ans

    # Kicks
    if "kick" in q or "kicks" in q or "geri" in q:
        ans = _answer_kicks_list(records)
        if ans:
            return ans

    # Blocks
    if "block" in q or "blocks" in q or "blocking" in q:
        ans = _answer_blocks_list(records)
        if ans:
            return ans

    # Uke Nagashi
    if "uke nagashi" in q:
        ans = _answer_uke_nagashi(records)
        if ans:
            return ans

    # Principles
    if "principles of striking" in q or ("principles" in q and "striking" in q):
        ans = _answer_principles(records)
        if ans:
            return ans

    return None


def _answer_specific_daken(question: str, records: Dict[str, Dict[str, str]]) -> Optional[str]:
    """
    Try to match a specific strike/block/kick name in the question.
    This is used when the user mentions Dakentaijutsu or clearly
    means striking, not general techniques.
    """
    q = _fold(question)
    for key, rec in records.items():
        if not key:
            continue
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, q):
            name = rec["name"]
            desc = rec["desc"]
            cat = rec.get("category") or "Dakentaijutsu"
            if desc:
                return f"{name} ({cat}): {desc}"
            else:
                return f"{name} ({cat})."
    return None


# ----------------- public entrypoint -----------------


def try_answer_dakentaijutsu(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic Dakentaijutsu extractor:

      * Lists:
          - 'what are the Hoken Juroppo Ken?'
          - 'list the dakentaijutsu kicks'
          - 'what are the basic blocks in dakentaijutsu?'
          - 'what is Uke Nagashi?'

      * Specific names (when context clearly about striking):
          - 'in dakentaijutsu, what is Jodan Uke?'
          - 'explain Ken Kudaki in dakentaijutsu'
    """
    if not _looks_like_daken_question(question):
        return None

    records, hoken_desc = _parse_daken_records()
    if not records:
        return None

    # 1) List-style questions (Hoken, kicks, blocks, uke nagashi, principles)
    ans = _answer_list_style(question, records, hoken_desc)
    if ans:
        return ans

    # 2) Specific named strike/block/kick
    ans = _answer_specific_daken(question, records)
    if ans:
        return ans

    return None
