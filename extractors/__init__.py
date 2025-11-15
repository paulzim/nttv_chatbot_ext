# extractors/__init__.py
from typing import List, Dict, Any, Optional

# ----------------------------------------------------------------------
# 1) Rank-specific extractors (most precise; run first)
# ----------------------------------------------------------------------
from .rank import (
    try_answer_rank_striking,
    try_answer_rank_nage,
    try_answer_rank_jime,
    try_answer_rank_requirements,  # explicit “requirements for X kyu”
)

# Optional: rank → weapons mapping (if present in rank.py)
try:
    from .rank import try_answer_rank_weapons  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    def try_answer_rank_weapons(
        question: str, passages: List[Dict[str, Any]]
    ) -> Optional[str]:
        return None


# ----------------------------------------------------------------------
# 2) Weapons (profiles, rank, katana parts)
# ----------------------------------------------------------------------
from .weapons import (
    try_answer_weapon_rank,
    try_answer_weapon_profile,
    try_answer_katana_parts,
)

# ----------------------------------------------------------------------
# 3) Kamae, Kyusho, Kihon Happo, Sanshin, Etiquette
# ----------------------------------------------------------------------
from .kamae import try_answer_kamae
from .kyusho import try_answer_kyusho

# IMPORTANT: use the real exported name from kihon_happo.py
from .kihon_happo import try_answer_kihon_happo

from .sanshin import try_answer_sanshin
from .etiquette import try_answer_etiquette

# ----------------------------------------------------------------------
# 4) Schools, techniques, leadership
# ----------------------------------------------------------------------
# IMPORTANT: use the real exported name from schools.py
from .schools import try_answer_school_profile as try_answer_schools

from .techniques import try_answer_technique
from .leadership import try_extract_answer as try_leadership

# ----------------------------------------------------------------------
# 5) Curriculum block extractors (Taihen, Dakentaijutsu, Nage, Jime, Gyaku)
#    These are optional – we guard imports so pytest doesn’t explode if any
#    of the files aren’t present locally.
# ----------------------------------------------------------------------
try:
    from .taihenjutsu import try_answer_taihenjutsu
except Exception:  # pragma: no cover
    def try_answer_taihenjutsu(
        question: str, passages: List[Dict[str, Any]]
    ) -> Optional[str]:
        return None


try:
    from .dakentaijutsu import try_answer_dakentaijutsu
except Exception:  # pragma: no cover
    def try_answer_dakentaijutsu(
        question: str, passages: List[Dict[str, Any]]
    ) -> Optional[str]:
        return None


try:
    from .nage_waza import try_answer_nage_waza
except Exception:  # pragma: no cover
    def try_answer_nage_waza(
        question: str, passages: List[Dict[str, Any]]
    ) -> Optional[str]:
        return None


try:
    from .jime_waza import try_answer_jime_waza
except Exception:  # pragma: no cover
    def try_answer_jime_waza(
        question: str, passages: List[Dict[str, Any]]
    ) -> Optional[str]:
        return None


try:
    from .gyaku_waza import try_answer_gyaku_waza
except Exception:  # pragma: no cover
    def try_answer_gyaku_waza(
        question: str, passages: List[Dict[str, Any]]
    ) -> Optional[str]:
        return None


# ----------------------------------------------------------------------
# Master router
# ----------------------------------------------------------------------
def try_extract_answer(
    question: str, passages: List[Dict[str, Any]]
) -> Optional[str]:
    """
    Master deterministic router for the NTTV chatbot.

    Given a natural-language question and a list of RAG passages, attempt to
    answer using the more specific extractors first, then fall back to more
    general ones. Returns a string answer or None if nothing deterministic
    applies (handing control back to the model).

    Order here is *very* intentional:
      1) Rank-specific queries
      2) Curriculum block queries (Taihen, Dakentaijutsu, Nage, Jime, Gyaku)
      3) Weapons (katana parts, profiles, rank)
      4) Kamae / Kyusho
      5) Kihon Happo / Sanshin / Etiquette
      6) Techniques / Schools
      7) Leadership
    """

    # ------------------------------------------------------------------
    # 1) Rank-specific logic (most constrained, very explicit intent)
    # ------------------------------------------------------------------
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

    # “What are the requirements for 8th kyu?”
    ans = try_answer_rank_requirements(question, passages)
    if ans:
        return ans

    # “What weapons do we learn at 3rd kyu?” (if implemented)
    ans = try_answer_rank_weapons(question, passages)
    if ans:
        return ans

    # ------------------------------------------------------------------
    # 2) Curriculum blocks (Taihenjutsu / Dakentaijutsu / Nage / Jime / Gyaku)
    # ------------------------------------------------------------------
    # “What are the rolls in Taihenjutsu?”
    ans = try_answer_taihenjutsu(question, passages)
    if ans:
        return ans

    # “What strikes are in Dakentaijutsu?”
    ans = try_answer_dakentaijutsu(question, passages)
    if ans:
        return ans

    # “List the throws in Nage Waza”
    ans = try_answer_nage_waza(question, passages)
    if ans:
        return ans

    # “What chokes are in the curriculum?”
    ans = try_answer_jime_waza(question, passages)
    if ans:
        return ans

    # “What joint locks / gyaku waza are in the curriculum?”
    ans = try_answer_gyaku_waza(question, passages)
    if ans:
        return ans

    # ------------------------------------------------------------------
    # 3) Weapons: katana parts, profiles, rank
    # ------------------------------------------------------------------
    # Katana terminology / sword parts (from nttv training reference.txt)
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
    # “What are the kamae for 9th kyu?”, “what kamae with the hanbo?”
    ans = try_answer_kamae(question, passages)
    if ans:
        return ans

    # Kyusho pressure point questions
    ans = try_answer_kyusho(question, passages)
    if ans:
        return ans

    # ------------------------------------------------------------------
    # 5) Kihon Happo, Sanshin, Etiquette
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
