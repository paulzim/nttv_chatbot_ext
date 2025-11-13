import pathlib
from extractors import try_extract_answer

SCHOOLS = pathlib.Path("data") / "Schools of the Bujinkan Summaries.txt"

def _passages():
    return [{
        "text": SCHOOLS.read_text(encoding="utf-8"),
        "source": "Schools of the Bujinkan Summaries.txt",
        "meta": {"priority": 1},
    }]

def test_togakure_profile_has_translation_type_focus():
    q = "tell me about togakure ryu"
    ans = try_extract_answer(q, _passages())
    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()
    assert "translation" in low
    assert "type:" in low
    assert "ninjutsu" in low

def test_gyokko_profile_has_kosshijutsu():
    q = "tell me about gyokko ryu"
    ans = try_extract_answer(q, _passages())
    assert isinstance(ans, str) and ans.strip()
    assert "kosshi" in ans.lower()
