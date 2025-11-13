import pathlib
from extractors import try_extract_answer

RANK = pathlib.Path("data") / "nttv rank requirements.txt"
WEAP = pathlib.Path("data") / "NTTV Weapons Reference.txt"

def _passages():
    return [
        {"text": RANK.read_text(encoding="utf-8"), "source": "nttv rank requirements.txt", "meta": {"priority": 3}},
        {"text": WEAP.read_text(encoding="utf-8"), "source": "NTTV Weapons Reference.txt", "meta": {"priority": 1}},
    ]

def test_kusari_fundo_rank():
    q = "At what rank do I learn kusari fundo?"
    ans = try_extract_answer(q, _passages())
    assert isinstance(ans, str) and ans.strip()
    assert "4th kyu" in ans.lower()
