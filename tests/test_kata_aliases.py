from extractors.technique_aliases import expand_with_aliases, TECH_ALIASES


def test_expand_with_aliases_omote_gyaku_english_alias():
    out = expand_with_aliases("This drill uses a forward wrist lock.")
    # expand_with_aliases returns lowercased tokens
    assert "omote gyaku" in out


def test_expand_with_aliases_oni_kudaki_english_alias():
    out = expand_with_aliases("Practice ogre crusher from a grab.")
    assert "oni kudaki" in out


def test_expand_with_aliases_jumonji_short_name():
    out = expand_with_aliases("We start from Jumonji.")
    # The alias map contains "Jumonji no Kata" with "jumonji" as an alias
    assert "jumonji no kata" in out or "jumonji" in out


def test_tech_aliases_contains_expected_keys():
    # Sanity: the core canonical keys are still present
    for key in [
        "Omote Gyaku",
        "Ura Gyaku",
        "Musha Dori",
        "Ganseki Nage",
        "Jumonji no Kata",
        "Oni Kudaki",
    ]:
        assert key in TECH_ALIASES
