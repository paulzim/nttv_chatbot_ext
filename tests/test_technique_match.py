# tests/test_technique_match.py
from extractors.technique_match import is_single_technique_query, technique_name_variants, fold

def test_query_detection():
    assert is_single_technique_query("explain Ichimonji no Kata") == "Ichimonji no Kata"
    assert is_single_technique_query("define Ichimonji") == "Ichimonji"
    assert is_single_technique_query("what is Omote Gyaku?").lower().startswith("omote")

def test_variants_include_short_and_long():
    v = technique_name_variants("Jumonji no Kata")
    low = [fold(x) for x in v]
    assert fold("Jumonji") in low
    assert fold("Jumonji no Kata") in low

def test_variants_include_hyphen_space_swaps():
    v = technique_name_variants("Omote Gyaku")
    low = [fold(x) for x in v]
    assert fold("Omote-Gyaku") in low
    assert fold("Omote Gyaku") in low

def test_concept_queries_ignored():
    assert is_single_technique_query("what is the kihon happo?") is None
    assert is_single_technique_query("explain sanshin no kata") is None
    assert is_single_technique_query("what are the schools of the bujinkan") is None
