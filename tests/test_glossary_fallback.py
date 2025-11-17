import pathlib
from extractors.glossary import try_answer_glossary

GLOSS = pathlib.Path("data") / "Glossary - edit.txt"
TECH = pathlib.Path("data") / "Technique Descriptions.md"


def _gloss_passages():
    # Match the pattern used in other tests (e.g., schools, weapons)
    return [
        {
            "text": GLOSS.read_text(encoding="utf-8"),
            "source": "Glossary - edit.txt",
            "meta": {"priority": 1},
        }
    ]


def _gloss_and_tech_passages():
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


def test_glossary_happo_geri_definition():
    q = "What is Happo Geri?"
    ans = try_answer_glossary(q, _gloss_passages())

    assert isinstance(ans, str) and ans.strip()

    low = ans.lower()
    # Should echo the term and its meaning from the glossary
    assert "happo geri" in low
    assert "eight" in low  # "kicking in the eight directions" from the glossary


def test_glossary_short_term_query():
    # Very short, term-only query should also be handled
    q = "Happo Geri"
    ans = try_answer_glossary(q, _gloss_passages())

    assert isinstance(ans, str) and ans.strip()

    low = ans.lower()
    assert "happo geri" in low
    assert "eight" in low


def test_glossary_ignores_who_questions():
    # Glossary should NOT kick in for who/when/where style questions
    q = "Who is Hatsumi?"
    ans = try_answer_glossary(q, _gloss_passages())

    # Either None or empty/whitespace is acceptable for "no glossary answer"
    assert not ans


def test_glossary_backs_off_for_technique_like_query():
    # With technique descriptions available, technique-like questions
    # should not be answered by the glossary.
    q = "Describe Oni Kudaki"
    ans = try_answer_glossary(q, _gloss_and_tech_passages())

    assert not ans


def test_glossary_backs_off_for_short_technique_name():
    # Even a short technique name alone should not be hijacked by the glossary
    # when it exists in Technique Descriptions.
    q = "Oni Kudaki"
    ans = try_answer_glossary(q, _gloss_and_tech_passages())

    assert not ans
