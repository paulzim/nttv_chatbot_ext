from pathlib import Path

from extractors import try_extract_answer

DATA = Path("data")

RANK = DATA / "nttv rank requirements.txt"
WEAPONS = DATA / "NTTV Weapons Reference.txt"
GLOSS = DATA / "Glossary - edit.txt"
TECH = DATA / "Technique Descriptions.md"
SANSHIN = DATA / "nttv training reference.txt"


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

def _passages_sanshin_and_gloss():
    return [
        {
            "text": SANSHIN.read_text(encoding="utf-8"),
            "source": "nttv training reference.txt",
            "meta": {"priority": 2},
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
    assert "8th kyu kicks:" in low
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
    
    
    def test_router_handles_many_noisy_passages():
        """
       Performance guardrail: router should still return a correct answer
        when given a large number of irrelevant passages plus one real rank passage.
        We don't assert on timing; we just make sure it doesn't fall over
        or lose the correct result in the noise.
        """
    # 500+ noisy passages
    noisy = [
        {
            "text": "lorem ipsum dolor sit amet " * 5,
            "source": f"noise_{i}.txt",
            "meta": {"priority": 5},
        }
        for i in range(600)
    ]

    # Real rank data with proper priority
    passages = noisy + [
        {
            "text": RANK.read_text(encoding="utf-8"),
            "source": "nttv rank requirements.txt",
            "meta": {"priority": 3},
        }
    ]

    q = "What kicks do I need to know for 8th kyu?"
    ans = try_extract_answer(q, passages)

    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()

    # Same expectations as our normal rank test: we got the right answer,
    # not lost in the noise.
    assert "8th kyu kicks:" in low
    assert "sokuho geri" in low
    assert "koho geri" in low

def test_router_musha_dori_meaning_phrase():
    """
    'What does Musha Dori mean?' currently does not have a deterministic
    extractor answer and should fall through to the LLM path (None here).
    This test protects that behavior from accidentally being hijacked by
    an overly eager glossary or rank extractor.
    """
    passages = [
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

    q = "What does Musha Dori mean?"
    ans = try_extract_answer(q, passages)

    # No deterministic answer yet → must be None/empty
    assert not ans



def test_router_explain_musha_dori_phrase():
    """
    'Explain Musha Dori' should also go through the technique extractor.
    """
    passages = [
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

    q = "Explain Musha Dori"
    ans = try_extract_answer(q, passages)

    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()

    assert "musha dori" in low
    assert "translation:" in low
    assert "definition:" in low


def test_router_tell_me_about_oni_kudaki():
    """
    'Tell me about Oni Kudaki' currently falls through to the LLM path
    (no deterministic answer). This test ensures we don't accidentally
    route it to an incorrect deterministic extractor.
    """
    passages = [
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

    q = "Tell me about Oni Kudaki"
    ans = try_extract_answer(q, passages)

    # No deterministic router answer expected here
    assert not ans



def test_router_list_sanshin_uses_sanshin_extractor():
    """
    'List Sanshin' should be answered by the Sanshin extractor, giving the
    five elemental Sanshin no Kata, not by the glossary.
    """
    passages = _passages_sanshin_and_gloss()

    q = "List Sanshin"
    ans = try_extract_answer(q, passages)

    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()

    # Should clearly be about Sanshin no Kata
    assert "sanshin no kata" in low or "san shin no kata" in low

    # And list the five elemental forms
    for tok in [
        "chi no kata",
        "sui no kata",
        "ka no kata",
        "fu no kata",
        "ku no kata",
    ]:
        assert tok in low


def test_router_diff_omote_vs_ura_gyaku():
    """
    'What's the difference between Omote Gyaku and Ura Gyaku?'
    should be handled by the technique diff extractor, producing
    a structured comparison derived from Technique Descriptions.
    """
    passages = [
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

    q = "What's the difference between Omote Gyaku and Ura Gyaku?"
    ans = try_extract_answer(q, passages)

    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()

    # Both techniques should be present
    assert "omote gyaku" in low
    assert "ura gyaku" in low

    # Diff extractor should emit "difference between ..."
    assert "difference between" in low

    # And show structured fields
    assert "translation:" in low
    assert "type:" in low
    assert "description:" in low
