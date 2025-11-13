import pathlib
from extractors.schools import try_answer_school_profile

SCHOOLS = pathlib.Path("data") / "Schools of the Bujinkan Summaries.txt"


def _passages():
    return [
        {
            "text": SCHOOLS.read_text(encoding="utf-8"),
            "source": "Schools of the Bujinkan Summaries.txt",
            "meta": {"priority": 1},
        }
    ]


def test_togakure_profile_has_translation_type_focus():
    q = "tell me about togakure ryu"
    ans = try_answer_school_profile(q, _passages())
    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()
    assert "togakure ryu" in low
    assert "translation" in low
    assert "type" in low
    assert "focus" in low


def test_gyokko_profile_mentions_kosshijutsu():
    q = "tell me about gyokko ryu"
    ans = try_answer_school_profile(q, _passages())
    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()
    assert "gyokko ryu" in low
    assert "kosshi" in low  # kosshijutsu focus
