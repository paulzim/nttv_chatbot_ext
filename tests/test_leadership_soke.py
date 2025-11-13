import os
from extractors import try_extract_answer

def _passages():
    # Minimal passages for leadership extractor to see the leadership doc
    with open(os.path.join("data", "Bujinkan Leadership and Wisdom.txt"), "r", encoding="utf-8") as f:
        txt = f.read()
    return [{"text": txt, "source": "Bujinkan Leadership and Wisdom.txt", "meta": {"priority": 1}}]

def test_gyokko_ryu_soke():
    q = "who is the soke of gyokko ryu?"
    ans = try_extract_answer(q, _passages())
    assert isinstance(ans, str) and ans.strip()
    low = ans.lower()
    # Prefer new truth, but allow backward-compatible acceptance if a user’s file lags
    assert ("ishizuka" in low) or ("nagato" in low)
