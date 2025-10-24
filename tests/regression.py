def test_rank_kicks_whitelist():
    from extractors.rank import try_rank
    rank_blob = """=== 8th Kyu ===
Striking:
  - Hoken Juroppo Ken; Sokuho Geri; Koho Geri; Sakui Geri; Happo Geri
"""
    hits = [{"text": rank_blob, "meta": {"priority": 3}}]
    ans = try_rank("What are the kicks for 8th kyu?", hits).lower()
    assert "sokuho geri" in ans and "happo geri" in ans
    assert "hoken juroppo ken" not in ans

def test_schools_trigger():
    from extractors.schools import try_schools
    blob = "The schools of the Bujinkan are: Togakure-ryū, Gyokushin-ryū, Kumogakure-ryū, Gyokko-ryū..."
    out = try_schools("List the Bujinkan schools", [{"text": blob}])
    assert out and "Togakure" in out
