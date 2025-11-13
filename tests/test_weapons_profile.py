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
    assert "hanbo" in low or "hanbō" in low
    assert "type:" in low
    assert "core actions" in low


def test_kusari_fundo_profile_mentions_chain_and_core_actions():
    q = "Explain the kusari fundo weapon."
    ans = try_answer_weapon_profile(q, _passages())
    assert isinstance(ans, str) and ans.strip()

    low = ans.lower()
    assert "kusari" in low
    assert "chain" in low
    assert "core actions" in low
