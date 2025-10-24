# tests/test_rank_prompts.py
import os
import re
import glob
import pytest

# Import your extractor router
# Assumes extractors/__init__.py exposes try_extract_answer(question, passages)
from extractors import try_extract_answer

ROOT = os.path.dirname(os.path.dirname(__file__))  # repo/tests -> repo
RANK_PATH = os.path.join(ROOT, "data", "nttv rank requirements.txt")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

def _read_rank_text() -> str:
    with open(RANK_PATH, "r", encoding="utf-8") as f:
        return f.read()

def _load_prompt_blocks(path: str):
    """
    Parse a simple key:value prompt file with blocks separated by lines '---'.
    Supported keys: QUESTION, EXPECT_ALL, EXPECT_ANY, EXPECT_NOT
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    blocks = [b.strip() for b in re.split(r"^\s*---\s*$", raw, flags=re.M) if b.strip()]
    cases = []
    for b in blocks:
        obj = {"QUESTION": "", "EXPECT_ALL": [], "EXPECT_ANY": [], "EXPECT_NOT": []}
        for line in b.splitlines():
            if not line.strip():
                continue
            if ":" not in line:
                # ignore malformed lines; keep format forgiving
                continue
            k, v = line.split(":", 1)
            key = k.strip().upper()
            val = v.strip()
            if key == "QUESTION":
                obj["QUESTION"] = val
            elif key == "EXPECT_ALL":
                obj["EXPECT_ALL"] = [t.strip() for t in val.split(",") if t.strip()]
            elif key == "EXPECT_ANY":
                obj["EXPECT_ANY"] = [t.strip() for t in re.split(r"\|", val) if t.strip()]
            elif key == "EXPECT_NOT":
                obj["EXPECT_NOT"] = [t.strip() for t in val.split(",") if t.strip()]
        if obj["QUESTION"]:
            cases.append(obj)
    return cases

def _collect_cases():
    files = sorted(glob.glob(os.path.join(PROMPTS_DIR, "*.txt")))
    all_cases = []
    for fp in files:
        for c in _load_prompt_blocks(fp):
            all_cases.append((os.path.basename(fp), c))
    return all_cases

@pytest.mark.parametrize("source_file,case", _collect_cases())
def test_rank_prompt_cases(source_file, case):
    question = case["QUESTION"]
    rank_text = _read_rank_text()

    # The first passage is the injected rank file (priority doc)
    passages = [{
        "text": rank_text,
        "source": "nttv rank requirements.txt",
        "meta": {"priority": 3},
    }]

    # Call the deterministic extractor router
    answer = try_extract_answer(question, passages)

    assert isinstance(answer, str) and answer.strip(), f"No answer for: {question}"

    ans_lo = answer.lower()

    # All required tokens must appear
    for tok in case.get("EXPECT_ALL", []):
        assert tok.lower() in ans_lo, f"Missing token '{tok}' in answer: {answer}"

    # At least one of EXPECT_ANY must appear
    any_list = case.get("EXPECT_ANY", [])
    if any_list:
        assert any(tok.lower() in ans_lo for tok in any_list), \
            f"None of EXPECT_ANY tokens {any_list} found in answer: {answer}"

    # Ensure forbidden tokens do not appear
    for tok in case.get("EXPECT_NOT", []):
        assert tok.lower() not in ans_lo, f"Forbidden token '{tok}' present in answer: {answer}"
