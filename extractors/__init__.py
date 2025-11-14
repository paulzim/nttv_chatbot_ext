# extractors/__init__.py
from typing import List, Dict, Any, Optional

# ----- Rank-specific extractors (most precise; run first)
from .rank import (
    try_answer_rank_striking,
    try_answer_rank_nage,
    try_answer_rank_jime,
    try_answer_rank_requirements,  # explicit “requirements for X kyu”
)

# If you implemented rank weapons in rank.py, import it; otherwise provide a no-op.
try:
    from .rank import try_answer_rank_weapons  # optional
except Exception:  # pragma: no cover
    def try_answer_rank_weapons(
        question: str, passages: List[Dict[str, Any]]
    ) -> Optional[str]:
        return None


# Weapons: rank, profiles, katana parts live here
from .weapons import (
    try_answer_weapon_rank,
    try_answer_weapon_profile,
    try_answer_katana_parts,
)

# Kamae (stances for empty-hand and weapons)
from .kamae import try_answer_kamae

# Kyusho (pressure points)
from .kyusho import try_answer_kyusho

# Kihon Happo (broken down by families, kata names, etc.)
from .kihon_happo import try_extract_answer as try_answer_kihon_happo

# Schools (Gyokko-ryu, Koto-ryu, Togakure-ryu, etc.)
from .schools import try_extract_answer as try_answer_schools

# Technique CSV / single-technique descriptions
from .techniques import try_answer_technique

# Sanshin no Kata
from .sanshin import try_answer_sanshin

# Leadership / sōke / shihan info
from .leadership import try_extract_answer as try_leadership

# Taihenjutsu (ukemi, rolls, breakfalls, etc.)
from .taihenjutsu import try_answer_taihenjutsu

# Dakentaijutsu (striking curriculum)
from .dakentaijutsu import try_answer_dakentaijutsu

# Nage waza (throws)
from .nage_waza import try_answer_nage_waza

# Jime waza (chokes / strangles)
from .jime_waza import try_answer_jime_waza

# Gyaku waza (reversals / joint locks)
from .gyaku_waza import try_answer_gyaku_waza

# Dojo etiquette / zanshin / dojo Japanese
from .etiquette import try_answer_etiquette


def try_extract_answer(
    question: str, passages: List[Dict[str, Any]]
) -> Optional[str]:
    """
    Master deterministic router for the NTTV chatbot.

    Given a natural-language question and a list of RAG passages, attempt to
    answer using the more specific extractors first, then fall back to more
    general ones. Returns a string answer or None if nothing deterministic
    applies (handing control back to the model).
    """

    # ------------------------------------------------------------------
    # 1) Rank-specific logic (most constrained, very explicit intent)
    # ------------------------------------------------------------------
    # “What are the requirements for 8th kyu?”
    ans = try_answer_rank_requirements(question, passages)
    if ans:
        return ans

    # “What punches are at 9th kyu?”, “what throws are required at 5th kyu?”
    ans = try_answer_rank_striking(question, passages)
    if ans:
        return ans

    ans = try_answer_rank_nage(question, passages)
    if ans:
        return ans

    ans = try_answer_rank_jime(question, passages)
    if ans:
        return ans

    # “What weapons do we learn at 3rd kyu?” (if implemented)
    ans = try_answer_rank_weapons(question, passages)
    if ans:
        return ans

    # ------------------------------------------------------------------
    # 2) Curriculum blocks (Taihenjutsu / Dakentaijutsu / Nage / Jime / Gyaku)
    # ------------------------------------------------------------------
    # “What are the rolls in Taihenjutsu?”, etc.
    ans = try_answer_taihenjutsu(question, passages)
    if ans:
        return ans

    # “What strikes are in Dakentaijutsu?”, etc.
    ans = try_answer_dakentaijutsu(question, passages)
    if ans:
        return ans

    # “List the throws in Nage waza”, “which throws are in the curriculum?”
    ans = try_answer_nage_waza(question, passages)
    if ans:
        return ans

    # “What chokes are in the curriculum?”, “what jime waza do we study?”
    ans = try_answer_jime_waza(question, passages)
    if ans:
        return ans

    # “What joint locks are in the curriculum?”, “what gyaku waza do we use?”
    ans = try_answer_gyaku_waza(question, passages)
    if ans:
        return ans

    # ------------------------------------------------------------------
    # 3) Weapons: katana parts, profiles, rank
    # ------------------------------------------------------------------
    # Katana terminology / parts of the sword
    ans = try_answer_katana_parts(question, passages)
    if ans:
        return ans

    # General weapon profile questions:
    # “What is the hanbo weapon?”, “what are the types of shuriken?”
    ans = try_answer_weapon_profile(question, passages)
    if ans:
        return ans

    # Weapon rank questions:
    # “At what rank do we learn kusari fundo?”
    ans = try_answer_weapon_rank(question, passages)
    if ans:
        return ans

    # ------------------------------------------------------------------
    # 4) Kamae (stances) and Kyusho (pressure points)
    # ------------------------------------------------------------------
    # “What are the kamae for 9th kyu?”, “what kamae with hanbo?”, etc.
    ans = try_answer_kamae(question, passages)
    if ans:
        return ans

    # Kyusho (pressure point questions)
    ans = try_answer_kyusho(question, passages)
    if ans:
        return ans

    # ------------------------------------------------------------------
    # 5) Kihon Happo, Sanshin, and Etiquette
    # ------------------------------------------------------------------
    # Kihon Happo breakdowns, families, and kata-level questions
    ans = try_answer_kihon_happo(question, passages)
    if ans:
        return ans

    # Sanshin no Kata (Chi / Sui / Ka / Fu / Ku)
    ans = try_answer_sanshin(question, passages)
    if ans:
        return ans

    # Dojo etiquette, bowing in, zanshin, dojo Japanese, counting
    ans = try_answer_etiquette(question, passages)
    if ans:
        return ans

    # ------------------------------------------------------------------
    # 6) Single-technique descriptions & school profiles
    # ------------------------------------------------------------------
    # Individual techniques from Technique Descriptions.md
    ans = try_answer_technique(question, passages)
    if ans:
        return ans

    # School profiles: “tell me about Togakure-ryu”, etc.
    ans = try_answer_schools(question, passages)
    if ans:
        return ans

    # ------------------------------------------------------------------
    # 7) Leadership / Sōke / shihan (lineage)
    # ------------------------------------------------------------------
    ans = try_leadership(question, passages)
    if ans:
        return ans

    # Nothing deterministic applied; leave it to the model.
    return None
