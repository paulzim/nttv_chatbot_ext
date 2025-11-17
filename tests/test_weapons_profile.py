import pathlib
from extractors.weapons import try_answer_weapon_profile

WEAP = pathlib.Path("data") / "NTTV Weapons Reference.txt"


def _passages():
    return [
        {
            "text": WEAP.read_text(encoding="utf-8"),
            "source": "NTTV Weapons Reference.txt",
            "meta": {"priority": 1},
        }
    ]


def test_hanbo_profile_uses_structured_fields():
    q = "What is the hanbo weapon?"
    ans = try_answer_weapon_profile(q, _passages())
    assert isinstance(ans, str) and ans.strip()

    low = ans.lower()
    # Name + type description pulled from the structured reference
    assert "hanbo" in low or "hanbō" in low
    assert "short staff" in low
    # And it should be using the CORE ACTIONS field
    assert "core actions include" in low


def test_kusari_fundo_profile_mentions_chain_and_core_actions():
    q = "Explain the kusari fundo weapon."
    ans = try_answer_weapon_profile(q, _passages())
    assert isinstance(ans, str) and ans.strip()

    low = ans.lower()
    assert "kusari" in low
    assert "chain" in low  # flexible chain weapon
    assert "core actions include" in low


from pathlib import Path
from extractors.weapons import try_answer_weapon_profile, try_answer_katana_parts

WEAPONS = Path("data") / "NTTV Weapons Reference.txt"


def _passages_weapons_only():
    return [
        {
            "text": WEAPONS.read_text(encoding="utf-8"),
            "source": "NTTV Weapons Reference.txt",
            "meta": {"priority": 1},
        }
    ]


def test_hanbo_weapon_profile_basic_fields():
    q = "What is a hanbo weapon?"
    ans = try_answer_weapon_profile(q, _passages_weapons_only())

    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()

    assert "hanbo weapon profile:" in low
    assert "type:" in low
    assert "core actions include:" in low
    assert "ranks:" in low
