import pathlib, re
from extractors import try_extract_answer

TECH = pathlib.Path("data") / "Technique Descriptions.md"

def _passages():
    return [{"text": TECH.read_text(encoding="utf-8"), "source": "Technique Descriptions.md", "meta": {"priority": 1}}]

def test_omote_gyaku_fields_present():
    q = "what is Omote Gyaku"
    ans = try_extract_answer(q, _passages())
    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()
    # basic structured fields from the technique extractor
    assert "english" in low and "family" in low and "rank intro" in low and "definition" in low
    assert "wrist" in low or "joint" in low
