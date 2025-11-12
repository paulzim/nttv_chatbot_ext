# tests/test_kata_aliases.py
from extractors.technique_aliases import expand_with_aliases

def test_expand_short_to_no_kata():
    out = expand_with_aliases("Ichimonji")
    assert "Ichimonji" in out
    assert "Ichimonji no Kata" in out

def test_expand_no_kata_to_short():
    out = expand_with_aliases("Hicho no Kata")
    assert "Hicho" in out
    assert "Hicho no Kata" in out

def test_known_lock_variant():
    out = expand_with_aliases("Omote Gyaku")
    assert "Omote Gyaku" in out
    assert "Omote-Gyaku" in out

def test_sanshin_short_names():
    out = expand_with_aliases("Chi")
    assert "Chi" in out
    assert "Chi no Kata" in out
