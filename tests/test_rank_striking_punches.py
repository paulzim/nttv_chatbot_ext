import pathlib
from extractors import try_extract_answer

DATA = pathlib.Path("data") / "nttv rank requirements.txt"

def _read_rank_text():
    return DATA.read_text(encoding="utf-8")

def _passages_rank():
    return [{
        "text": _read_rank_text(),
        "source": "nttv rank requirements.txt",
        "meta": {"priority": 3},
    }]

def test_8th_kyu_kicks_rank_only():
    q = "What are the kicks for 8th kyu?"
    ans = try_extract_answer(q, _passages_rank())
    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()
    # Should NOT include 9th-kyu “front kick” unless cumulative phrasing is used
    assert "front kick" not in low and "zenpo geri" not in low and "mae geri" not in low
    # Should list 8th-kyu kicks present in source
    assert any(k in low for k in ["sokuho geri", "koho geri", "sakui geri", "happo geri"])

def test_8th_kyu_kicks_cumulative_from_need_to_know():
    q = "What are the kicks I need to know for 8th kyu?"
    ans = try_extract_answer(q, _passages_rank())
    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()
    # Cumulative should include 9th-kyu foundations
    assert any(k in low for k in ["zenpo geri", "mae geri", "front kick"])

def test_4th_kyu_throws():
    q = "What are the throws for 4th kyu?"
    ans = try_extract_answer(q, _passages_rank())
    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()
    assert any(tok in low for tok in ["osoto", "oosoto", "seoi", "nage"])

def test_3rd_kyu_chokes():
    q = "What are the chokes for 3rd kyu?"
    ans = try_extract_answer(q, _passages_rank())
    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()
    # Expect “jime” family names for 3rd kyu
    assert "jime" in low
