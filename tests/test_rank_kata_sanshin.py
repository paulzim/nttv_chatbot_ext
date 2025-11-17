import pathlib
from extractors import try_extract_answer

DATA = pathlib.Path("data") / "nttv rank requirements.txt"


def _passages_rank():
    return [{
        "text": DATA.read_text(encoding="utf-8"),
        "source": "nttv rank requirements.txt",
        "meta": {"priority": 3},
    }]


def test_8th_kyu_kihon_happo_kata():
    q = "Which Kihon Happo kata are required for 8th kyu?"
    ans = try_extract_answer(q, _passages_rank())
    assert isinstance(ans, str) and ans.strip()

    low = ans.lower()
    assert "8th kyu" in low
    assert "kihon happo" in low
    # From data/nttv rank requirements.txt for 8th kyu
    assert "ichimonji no kata" in low


def test_8th_kyu_sanshin_kata():
    q = "What Sanshin no Kata do I need for 8th kyu?"
    ans = try_extract_answer(q, _passages_rank())
    assert isinstance(ans, str) and ans.strip()

    low = ans.lower()
    assert "8th kyu" in low
    assert ("san shin no kata" in low) or ("sanshin no kata" in low)

    # From data/nttv rank requirements.txt for 8th kyu
    for tok in [
        "chi no kata",
        "sui no kata",
        "ka no kata",
        "fu no kata",
        "ku no kata",
    ]:
        assert tok in low

def test_8th_kyu_kihon_happo_without_word_kata():
    q = "What Kihon Happo do I need to know for 8th kyu?"
    ans = try_extract_answer(q, _passages_rank())
    assert isinstance(ans, str) and ans.strip()

    low = ans.lower()
    assert "8th kyu" in low
    assert "kihon happo" in low
    assert "ichimonji no kata" in low

def test_kata_queries_without_rank_use_generic_kihon_description():
    """
    If the question mentions Kihon Happo but NO rank,
    we should return the generic Kihon Happo description
    (not a rank-specific answer).
    """
    q = "Which Kihon Happo kata are there?"
    ans = try_extract_answer(q, _passages_rank())

    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()

    # Should talk about Kihon Happo in general terms
    assert "kihon happo" in low
    assert "kosshi kihon sanpo" in low
    assert "torite goho" in low

    # Should NOT pin to a specific rank like "8th kyu"
    assert "8th kyu" not in low
    assert "7th kyu" not in low
    assert "9th kyu" not in low
