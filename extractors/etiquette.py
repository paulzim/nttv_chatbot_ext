from __future__ import annotations
from typing import List, Dict, Any, Optional
import unicodedata
import re


def _fold(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()


# ----------------- canned content -----------------

BOW_IN_TEXT = (
    "Bow-in procedure (as taught in this curriculum) includes:\n"
    "• Arriving with enough time to be on the mat and ready when class begins.\n"
    "• Lining up in rank order facing the front (shomen) of the dojo.\n"
    "• Bowing together toward the shomen, then toward the instructor, and then toward your training partners.\n"
    "• Using dojo phrases like “Onegaishimasu” at the start (“please train with me”) and "
    "“Domo arigato gozaimashita” at the end (“thank you very much”).\n"
    "The exact form can vary by dojo, but you should be able to participate smoothly and respectfully."
)

BOW_LATE_TEXT = (
    "Bowing in late (dojo etiquette):\n"
    "• If you arrive late, wait at the edge of the mat until the instructor acknowledges you.\n"
    "• When given permission, bow toward the shomen and the instructor before stepping onto the mat.\n"
    "• Join the line or your training partner quietly without disrupting ongoing practice.\n"
    "• After class, apologize briefly if needed — the main point is to show respect and minimize disruption."
)

DOJO_JAPANESE_TEXT = (
    "Basic dojo Japanese for this curriculum (9th Kyu):\n"
    "• Onegaishimasu — “Please assist me” (said at the start of training or paired practice).\n"
    "• Domo arigato gozaimashita — “Thank you very much” (often at the end of class).\n"
    "• Shiken Haramitsu Daikomyo — “Through every encounter, may we be brought to the highest light.”\n"
    "• Yame — “Stop.”\n"
    "• Counting from 1 to 10 in Japanese: ichi, ni, san, yon/shi, go, roku, nana/shichi, hachi, kyuu/ku, juu.\n"
    "These phrases and counting are part of basic dojo etiquette and should be used sincerely, not mechanically."
)

COUNTING_TEXT = (
    "Counting from 1 to 10 in Japanese (as used in class):\n"
    "1 — ichi (いち)\n"
    "2 — ni (に)\n"
    "3 — san (さん)\n"
    "4 — yon / shi (よん / し)\n"
    "5 — go (ご)\n"
    "6 — roku (ろく)\n"
    "7 — nana / shichi (なな / しち)\n"
    "8 — hachi (はち)\n"
    "9 — kyuu / ku (きゅう / く)\n"
    "10 — juu (じゅう)\n"
    "You will often hear these used to count reps during basics, ukemi, or conditioning drills."
)

ZANSHIN_BEGINNER_TEXT = (
    "Zanshin (awareness) at the beginner level in this curriculum includes:\n"
    "• Keep your mouth closed — stay focused, avoid unnecessary talking.\n"
    "• Keep your hands up — maintain a ready, protective posture even between drills.\n"
    "• Know who Masaaki Hatsumi is — the Soke (grandmaster) of Bujinkan Budo Taijutsu.\n"
    "• Know who Toshitsugu Takamatsu was — Hatsumi’s teacher and previous generation master.\n"
    "This form of zanshin is about basic awareness, respect, and readiness in the dojo."
)

ZANSHIN_ADVANCED_TEXT = (
    "Zanshin (awareness) at higher levels in this curriculum adds:\n"
    "• Being able to perform techniques without directly staring at the opponent.\n"
    "• Training with one eye closed, both eyes closed, blindfolded, or under reduced vision.\n"
    "• Learning to feel timing, distance, and movement through peripheral cues, contact, and intent.\n"
    "• Noticing objects, people, and conditions in your environment even when you are focused on a technique.\n"
    "This advanced zanshin is about expanding awareness beyond the immediate target so you can move safely and effectively."
)


# ----------------- intent helpers -----------------


def _looks_like_etiquette_question(question: str) -> bool:
    q = _fold(question)
    return any(
        key in q
        for key in [
            "etiquette",
            "dojo etiquette",
            "bow in",
            "bowing in",
            "bow-in",
            "zanshin",
            "basic dojo japanese",
            "dojo japanese",
            "japanese phrases",
            "late to class",
            "arrive late",
            "coming in late",
            "count in japanese",
            "japanese numbers",
        ]
    )


def _wants_bow_in(question: str) -> bool:
    q = _fold(question)
    return ("bow in" in q or "bow-in" in q or "bowing in" in q) and "late" not in q


def _wants_bow_late(question: str) -> bool:
    q = _fold(question)
    return ("bow" in q and "late" in q) or "arrive late" in q or "coming in late" in q


def _wants_dojo_japanese(question: str) -> bool:
    q = _fold(question)
    return (
        "basic dojo japanese" in q
        or "dojo japanese" in q
        or "japanese phrases" in q
    )


def _wants_counting(question: str) -> bool:
    q = _fold(question)
    return (
        ("count" in q or "numbers" in q)
        and ("japanese" in q or "in japanese" in q)
    ) or "count in japanese" in q or "japanese numbers" in q


def _wants_zanshin(question: str) -> bool:
    q = _fold(question)
    return "zanshin" in q or ("awareness" in q and "zanshin" in q)


def _wants_advanced_zanshin(question: str) -> bool:
    q = _fold(question)
    return ("advanced" in q and "zanshin" in q) or ("higher level" in q and "zanshin" in q)


# ----------------- public entrypoint -----------------


def try_answer_etiquette(question: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic dojo etiquette extractor.

    Handles:
      * 'what is the bow in procedure?'
      * 'what do I do if I’m late to class?'
      * 'what is dojo etiquette at 9th kyu?'
      * 'what are the basic dojo Japanese phrases?'
      * 'how do you count in Japanese?'
      * 'what is zanshin?' / 'what is advanced zanshin?'
    """
    if not _looks_like_etiquette_question(question):
        return None

    if _wants_bow_late(question):
        return BOW_LATE_TEXT

    if _wants_bow_in(question):
        return BOW_IN_TEXT

    if _wants_counting(question):
        return COUNTING_TEXT

    if _wants_dojo_japanese(question):
        # Include phrases + a brief counting mention
        return DOJO_JAPANESE_TEXT

    if _wants_zanshin(question):
        # If they explicitly ask for advanced, show that version
        if _wants_advanced_zanshin(question):
            return ZANSHIN_ADVANCED_TEXT
        # General 'what is zanshin?'
        return ZANSHIN_BEGINNER_TEXT + "\n\n" + ZANSHIN_ADVANCED_TEXT

    # Generic etiquette question
    q = _fold(question)
    if "etiquette" in q or "dojo etiquette" in q:
        return (
            "Dojo etiquette at 9th Kyu in this curriculum includes:\n"
            "• Learning the bow-in procedure so you can line up, bow to shomen, bow to the instructor, and "
            "bow to your training partners correctly.\n"
            "• Knowing how to bow in respectfully if you arrive late to class.\n"
            "• Using basic dojo Japanese phrases such as “Onegaishimasu”, “Domo arigato gozaimashita”, "
            "“Shiken Haramitsu Daikomyo”, and “Yame”, and being able to count from 1 to 10 in Japanese during drills.\n"
            "• Practicing basic zanshin (awareness): keeping your mouth closed, hands up, and knowing who Hatsumi "
            "and Takamatsu are in the Bujinkan lineage.\n"
            "These are foundation-level etiquette skills expected of a new Bujinkan student."
        )

    # Fallback: if we matched the broad etiquette intent but not a subtype,
    # return a concise general etiquette summary.
    return (
        "This curriculum expects you to understand basic dojo etiquette: how to bow in, how to bow in late without "
        "disrupting class, how to use key Japanese phrases like “Onegaishimasu” and “Domo arigato gozaimashita”, "
        "how to count from 1 to 10 in Japanese during drills, and how to maintain zanshin (awareness) in the dojo."
    )
