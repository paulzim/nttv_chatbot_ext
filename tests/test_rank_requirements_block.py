import pathlib
from extractors import try_extract_answer

RANK = pathlib.Path("data") / "nttv rank requirements.txt"

def _passages():
    return [{
        "text": RANK.read_text(encoding="utf-8"),
        "source": "nttv rank requirements.txt",
        "meta": {"priority": 3},
    }]

def test_requirements_scoped_single_rank():
    q = "What are the rank requirements for 3rd kyu?"
    ans = try_extract_answer(q, _passages())
    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()
    # Must contain the requested header
    assert "3rd kyu" in low
    # Must not dump other rank headers
    for hdr in ["9th kyu", "8th kyu", "7th kyu", "6th kyu", "5th kyu", "4th kyu", "2nd kyu", "1st kyu", "shodan"]:
        if hdr != "3rd kyu":
            assert hdr not in low
