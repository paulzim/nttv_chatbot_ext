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


def test_oni_kudaki_definition_not_truncated_by_commas():
    """
    Guard against the original CSV-comma bug for Oni Kudaki.

    NOTE: The technique extractor currently strips the verb only for
    'what is', 'define', and 'explain', so we use 'Explain' here
    instead of 'Describe' to exercise the CSV path.
    """
    q = "Explain Oni Kudaki"
    ans = try_answer_technique(q, _passages())
    assert isinstance(ans, str) and ans.strip()

    first_line = ans.splitlines()[0].lower()
    assert "oni" in first_line and "kudaki" in first_line

    low = ans.lower()
    # Structured fields we expect from the technique extractor
    assert "translation" in low
    assert "type" in low
    assert "rank intro" in low
    assert "definition" in low

    # Make sure the definition wasn't chopped at the first comma
    def_lines = [ln for ln in ans.splitlines() if "definition" in ln.lower()]
    assert def_lines, "Expected a 'Definition:' line in Oni Kudaki answer"
    definition_text = def_lines[0].split(":", 1)[-1].strip()

    # If CSV splitting broke on the first comma, this will be very short
    assert len(definition_text) > 60, (
        f"Definition for Oni Kudaki looks suspiciously short: {definition_text!r}"
    )
