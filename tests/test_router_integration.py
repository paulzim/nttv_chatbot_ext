from pathlib import Path

from extractors import try_extract_answer

DATA = Path("data")

RANK = DATA / "nttv rank requirements.txt"
WEAPONS = DATA / "NTTV Weapons Reference.txt"
GLOSS = DATA / "Glossary - edit.txt"
TECH = DATA / "Technique Descriptions.md"


def _passages_rank_and_gloss():
    return [
        {
            "text": RANK.read_text(encoding="utf-8"),
            "source": "nttv rank requirements.txt",
            "meta": {"priority": 3},
        },
        {
            "text": GLOSS.read_text(encoding="utf-8"),
            "source": "Glossary - edit.txt",
            "meta": {"priority": 1},
        },
    ]


def _passages_weapons_and_gloss():
    return [
        {
            "text": WEAPONS.read_text(encoding="utf-8"),
            "source": "NTTV Weapons Reference.txt",
            "meta": {"priority": 1},
        },
        {
            "text": GLOSS.read_text(encoding="utf-8"),
            "source": "Glossary - edit.txt",
            "meta": {"priority": 1},
        },
    ]


def _passages_tech_and_gloss():
    return [
        {
            "text": TECH.read_text(encoding="utf-8"),
            "source": "Technique Descriptions.md",
            "meta": {"priority": 1},
        },
        {
            "text": GLOSS.read_text(encoding="utf-8"),
            "source": "Glossary - edit.txt",
            "meta": {"priority": 1},
        },
    ]


def test_router_prefers_rank_over_glossary_for_kicks():
    q = "What kicks do I need to know for 8th kyu?"
    ans = try_extract_answer(q, _passages_rank_and_gloss())

    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()

    # Should be the rank striking answer, not a glossary definition
    assert "8th kyu kicks:" in ans
    # sanity: shouldn't look like simple "Term: definition"
    assert not low.startswith("8th kyu:")


def test_router_prefers_weapon_profile_over_glossary():
    q = "What is a hanbo weapon?"
    ans = try_extract_answer(q, _passages_weapons_and_gloss())

    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()

    # Should use the weapons profile extractor
    assert "hanbo" in low
    assert "weapon profile:" in low
    assert "type:" in low  # from weapon profile formatting


def test_router_prefers_technique_over_glossary():
    q = "Describe Oni Kudaki"
    ans = try_extract_answer(q, _passages_tech_and_gloss())

    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()

    # Should be the structured technique answer, not glossary
    assert "oni kudaki:" in low
    assert "translation:" in low
    assert "definition:" in low
    # sanity: glossary-style "Oni Kudaki: Demon Crusher" alone isn't enough
    # we expect the bullet structure from the technique extractor


def test_router_glossary_fallback_when_no_specific_extractor():
    q = "What is Happo Geri?"
    # Only give it glossary; no rank/tech/weapon files
    passages = [
        {
            "text": GLOSS.read_text(encoding="utf-8"),
            "source": "Glossary - edit.txt",
            "meta": {"priority": 1},
        }
    ]
    ans = try_extract_answer(q, passages)

    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()

    assert "happo geri" in low
    assert "eight" in low  # from the glossary definition


def test_router_handles_many_noisy_passages_quickly():
    # A bunch of irrelevant passages plus the real rank data
    noisy = [
        {
            "text": "lorem ipsum dolor sit amet " * 5,
            "source": f"noise_{i}.txt",
            "meta": {"priority": 5},
        }
        for i in range(200)
    ]
    passages = noisy + [
        {
            "text": RANK.read_text(encoding="utf-8"),
            "source": "nttv rank requirements.txt",
            "meta": {"priority": 1},
        }
    ]

    q = "What kicks do I need to know for 8th kyu?"
    ans = try_extract_answer(q, passages)

    assert isinstance(ans, str) and ans.strip()
    assert "8th Kyu kicks:" in ans
