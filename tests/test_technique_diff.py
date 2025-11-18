import pathlib

from extractors.technique_diff import try_answer_technique_diff

TECH_FILE = pathlib.Path("data") / "Technique Descriptions.md"


def _passages_tech_only():
    return [
        {
            "text": TECH_FILE.read_text(encoding="utf-8"),
            "source": "Technique Descriptions.md",
            "meta": {"priority": 1},
        }
    ]


def test_diff_omote_vs_ura_gyaku_difference_between():
    q = "What is the difference between Omote Gyaku and Ura Gyaku?"
    ans = try_answer_technique_diff(q, _passages_tech_only())

    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()

    # Both techniques should be present
    assert "omote gyaku" in low
    assert "ura gyaku" in low

    # We should see some structured fields
    assert "translation:" in low
    assert "type:" in low
    assert "description:" in low


def test_diff_omote_vs_ura_gyaku_vs_syntax():
    q = "Omote Gyaku vs Ura Gyaku"
    ans = try_answer_technique_diff(q, _passages_tech_only())

    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()

    assert "omote gyaku" in low
    assert "ura gyaku" in low
    assert "translation:" in low


def test_non_diff_question_returns_none():
    q = "Describe Omote Gyaku"
    ans = try_answer_technique_diff(q, _passages_tech_only())

    # Not a diff/comparison question → no answer from this extractor
    assert not ans
