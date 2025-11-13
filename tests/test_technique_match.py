from extractors.technique_match import (
    is_single_technique_query,
    technique_name_variants,
    canonical_from_query,
    fold,
)


def test_single_technique_query_true_for_omote_gyaku():
    assert is_single_technique_query("explain Omote Gyaku") is True


def test_single_technique_query_false_for_concept():
    assert is_single_technique_query("what is the kihon happo?") is False


def test_variants_strip_no_kata_for_jumonji():
    v = technique_name_variants("Jumonji no Kata")
    folded = [fold(x) for x in v]
    assert "jumonji no kata" in folded
    assert "jumonji" in folded


def test_variants_collapse_hyphen_space_for_omote_gyaku():
    v = technique_name_variants("Omote-Gyaku")
    folded = [fold(x) for x in v]
    assert "omote gyaku" in folded


def test_canonical_from_english_alias_ogre_crusher():
    canon = canonical_from_query("Can you explain ogre crusher in the Bujinkan?")
    assert canon == "Oni Kudaki"
