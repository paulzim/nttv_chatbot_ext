# NTTV Chatbot — Tests

This test suite is designed to catch regressions in the deterministic extractors
(rank, schools, weapons, kihon happo, and technique lookups) without having to
click around in Streamlit.

## Quick start

1) Activate your venv and install test deps:
```bash
. .venv/Scripts/activate    # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Run the full suite:
```
```
bash
Copy code
pytest -q
Run a single file or test:
```
```
bash
Copy code
pytest -q tests/test_rank_prompts.py::test_rank_prompt_cases
pytest -q tests/test_schools_profile.py::test_togakure_profile_has_translation_type_focus
```
## What these tests cover
- Rank extractors
- Kicks/throws/chokes by rank (strict filtering to the asked rank)
- Rank “requirements for X kyu” block extraction
- Schools
- “List the nine schools of the Bujinkan”
- School profiles (translation / type / focus / weapons / notes)
- Weapons
- Weapon → first-rank introduction (from structured reference, falling back to training reference, then glossary)

- Short weapon profiles (type + a few key fields)

- Kihon Happo

- Canonical breakdown (Kosshi Kihon Sanpō + Torite Gohō) with stable wording

- Techniques

- Single-technique lookup (e.g., Omote Gyaku) and presence of core fields

## Troubleshooting
Module import errors in tests

Ensure the project root (folder containing extractors/) is on PYTHONPATH.

Easiest fix (Windows PowerShell):

powershell
Copy code
$env:PYTHONPATH = "$PWD"
pytest -q
Tests can’t see local data files

The tests are written to use the in-repo .md/.txt sources. If you renamed
or moved them, update the small loader helpers in the test files accordingly.

One failing test after extractor changes

Read the assertion message carefully. Most failures are either token mismatch
(normalize/alias a term) or too-broad matches (tighten the parsing regex or
filter step).

Conventions
Prefer deterministic extractors first (return str or None).

Keep extractors pure (no file I/O inside); tests provide passages.

Normalize macrons and punctuation before alias matching.

When formatting answers, keep them short and direct—tests look for key tokens.

Adding new tests
Copy an existing test file and adjust the input question + expected tokens.

Keep expectations resilient (use EXPECT_ANY token lists for synonyms).

If you add a new extractor, add at least one test that:

Asks a minimal question that should route to that extractor.

Verifies a couple of canonical tokens in the answer.


