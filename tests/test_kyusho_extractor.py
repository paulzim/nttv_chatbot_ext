import pathlib
from extractors.kyusho import try_answer_kyusho

KYU = pathlib.Path("data") / "KYUSHO.txt"


def _passages():
    return [
        {
            "text": KYU.read_text(encoding="utf-8"),
            "source": "KYUSHO.txt",
            "meta": {"priority": 1},
        }
    ]


def test_specific_point_ura_kimon_from_real_data():
    q = "Where is the Ura Kimon kyusho point?"
    ans = try_answer_kyusho(q, _passages())
    assert isinstance(ans, str) and ans.strip()

    low = ans.lower()
    assert "ura kimon" in low
    assert "ribs under the pectoral muscles" in low
