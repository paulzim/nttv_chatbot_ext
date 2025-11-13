import pathlib
from extractors.techniques import try_answer_technique

TECH = pathlib.Path("data") / "Technique Descriptions.md"


def _passages():
    return [
        {
            "text": TECH.read_text(encoding="utf-8"),
            "source": "Technique Descriptions.md",
            "meta": {"priority": 1},
        }
    ]


def test_omote_gyaku_fields_present():
    q = "what is Omote Gyaku"
    ans = try_answer_technique(q, _passages())
    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()
    # Structured fields from the technique extractor
    assert "translation" in low
    assert "type" in low
    assert "rank intro" in low
    assert "definition" in low
    assert "wrist" in low or "joint" in low
