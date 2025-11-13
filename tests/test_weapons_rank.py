import pathlib
from extractors.weapons import try_answer_weapon_rank

TRAIN = pathlib.Path("data") / "nttv training reference.txt"
WEAP = pathlib.Path("data") / "NTTV Weapons Reference.txt"


def _passages():
    return [
        {
            "text": WEAP.read_text(encoding="utf-8"),
            "source": "NTTV Weapons Reference.txt",
            "meta": {"priority": 1},
        },
        {
            "text": TRAIN.read_text(encoding="utf-8"),
            "source": "nttv training reference.txt",
            "meta": {"priority": 2},
        },
    ]


def test_kusari_fundo_rank():
    q = "At what rank do I learn kusari fundo?"
    ans = try_answer_weapon_rank(q, _passages())
    assert isinstance(ans, str) and ans.strip()
    assert "4th kyu" in ans.lower()
