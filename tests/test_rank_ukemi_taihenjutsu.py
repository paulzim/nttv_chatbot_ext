import pathlib

from extractors import try_extract_answer

RANK_FILE = pathlib.Path("data") / "nttv rank requirements.txt"


def _passages_rank_only():
    return [
        {
            "text": RANK_FILE.read_text(encoding="utf-8"),
            "source": "nttv rank requirements.txt",
            "meta": {"priority": 3},
        }
    ]


def test_9th_kyu_ukemi_list():
    q = "What ukemi do I need to know for 9th kyu?"
    ans = try_extract_answer(q, _passages_rank_only())

    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()

    assert "9th kyu" in low
    assert "ukemi" in low

    # From the 9th kyu Ukemi line
    assert "zenpo ukemi" in low
    assert "koho ukemi" in low
    assert "yoko ukemi" in low


def test_9th_kyu_ukemi_rolls_wording():
    q = "What rolls and breakfalls are required for 9th kyu?"
    ans = try_extract_answer(q, _passages_rank_only())

    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()

    assert "9th kyu" in low
    assert "ukemi" in low
    assert "zenpo ukemi" in low
    assert "koho ukemi" in low


def test_9th_kyu_taihenjutsu():
    q = "What taihenjutsu do I need to know for 9th kyu?"
    ans = try_extract_answer(q, _passages_rank_only())

    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()

    assert "9th kyu" in low
    assert "taihenjutsu" in low or "taihen jutsu" in low
    # From the 9th kyu Taihenjutsu line
    assert "tai sabaki" in low


def test_ukemi_without_rank_does_not_use_rank_specific():
    q = "What ukemi do I need to know?"
    ans = try_extract_answer(q, _passages_rank_only())

    # With only the rank doc, rank-specific extractor should not fire
    # (no kyu in the question), so this should return no deterministic answer.
    assert not ans
