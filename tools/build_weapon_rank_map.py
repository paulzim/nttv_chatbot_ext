# tools/build_weapon_rank_map.py
import json, re
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "nttv training reference.txt"
OUT_FILE  = Path(__file__).resolve().parents[1] / "index" / "weapon_rank_map.json"

RANK_HEAD = re.compile(r"^\s*(\d{1,2}(?:st|nd|rd|th)\s+kyu)\b", re.IGNORECASE)
WEAPONS_BLOCK_HEAD = re.compile(r"^\s*WEAPONS?\s*:\s*(.+?)\s*$", re.IGNORECASE)
BULLET = re.compile(r"^\s*[•\-\u2022]\s*(.+?)\s*$")

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _split_list(s: str):
    return [p.strip(" \t-•") for p in re.split(r"[;,/]| and ", s) if p.strip()]

def main():
    if not DATA_FILE.exists():
        raise SystemExit(f"Missing data file: {DATA_FILE}")
    text = DATA_FILE.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    out = {}
    current_rank = None
    i = 0
    while i < len(lines):
        rh = RANK_HEAD.match(lines[i])
        if rh:
            current_rank = rh.group(1).title()
            if current_rank not in out:
                out[current_rank] = []
            i += 1
            continue
        if current_rank:
            m = WEAPONS_BLOCK_HEAD.match(lines[i])
            if m:
                header_inline = _norm(m.group(1))
                if header_inline:
                    out[current_rank].extend(_split_list(header_inline))
                j = i + 1
                while j < len(lines):
                    ln = lines[j].strip()
                    if not ln:
                        break
                    if WEAPONS_BLOCK_HEAD.match(ln) or RANK_HEAD.match(ln):
                        break
                    b = BULLET.match(ln)
                    if b:
                        out[current_rank].extend(_split_list(_norm(b.group(1))))
                    j += 1
                i = j
                continue
        i += 1

    # tidy
    for rk, vals in out.items():
        seen, clean = set(), []
        for v in vals:
            vv = _norm(v)
            if vv and vv.lower() not in seen:
                seen.add(vv.lower()); clean.append(vv)
        out[rk] = clean

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT_FILE}")

if __name__ == "__main__":
    main()
