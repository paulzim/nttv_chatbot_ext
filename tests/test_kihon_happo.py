import pathlib
from extractors import try_extract_answer

RANK = pathlib.Path("data") / "nttv rank requirements.txt"
TECH = pathlib.Path("data") / "Technique Descriptions.md"


def _passages():
    return [
        {
            "text": RANK.read_text(encoding="utf-8"),
            "source": "nttv rank requirements.txt",
            "meta": {"priority": 3},
        },
        {
            "text": TECH.read_text(encoding="utf-8"),
            "source": "Technique Descriptions.md",
            "meta": {"priority": 1},
        },
    ]


def test_kihon_happo_answer_is_present_and_mentions_kata():
    q = "what is the kihon happo?"
    ans = try_extract_answer(q, _passages())
    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()
    assert "kihon happo" in low
    # Current extractor returns a short summary; make sure at least one kata appears
    assert "ichimonji" in low
