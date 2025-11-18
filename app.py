import os
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
import re

import numpy as np
import streamlit as st

# Vector index
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

# Embeddings
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None

# Deterministic extractors (dispatcher + specific modules)
from extractors.kihon_happo import try_answer_kihon_happo
from extractors import try_extract_answer
from extractors.leadership import try_extract_answer as try_leadership
from extractors.weapons import try_answer_weapon_rank
from extractors.rank import try_answer_rank_requirements
from extractors.schools import (
    try_answer_school_profile,
    try_answer_schools_list,   # list extractor
    SCHOOL_ALIASES,
    is_school_list_query,
)

from extractors.technique_match import (
    is_single_technique_query as _is_single_technique_query,
    technique_name_variants as _tech_name_variants,
)

# ---------------------------
# Load index & metadata
# ---------------------------
# Default: local ./index (for your LM Studio / local dev)
DEFAULT_INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")

# Allow Render (or any deployment) to override via env
INDEX_DIR = os.getenv("INDEX_DIR", DEFAULT_INDEX_DIR)
CONFIG_PATH = os.getenv("CONFIG_PATH", os.path.join(INDEX_DIR, "config.json"))
META_PATH = os.getenv("META_PATH", os.path.join(INDEX_DIR, "meta.pkl"))

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Missing index config: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

EMBED_MODEL_NAME = cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

# Env wins, then config, then local default
FAISS_PATH = (
    os.getenv("INDEX_PATH")
    or cfg.get("faiss_path")
    or os.path.join(INDEX_DIR, "faiss.index")
)

TOP_K = int(cfg.get("top_k", 6))

with open(META_PATH, "rb") as f:
    CHUNKS: List[Dict[str, Any]] = pickle.load(f)

if faiss is None:
    raise RuntimeError("faiss is not installed. Please `pip install faiss-cpu` (Windows: faiss-cpu).")

index = faiss.read_index(FAISS_PATH)


# Lazy-load embeddings
_EMBED_MODEL = None
def get_embedder():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed. `pip install sentence-transformers`")
        _EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    return _EMBED_MODEL

def embed_query(q: str) -> np.ndarray:
    model = get_embedder()
    v = model.encode([q], normalize_embeddings=True)
    return v.astype("float32")

# ---------------------------
# Retrieval & reranking
# ---------------------------
def retrieve(q: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    """Search FAISS, then rerank with filename priority, query-aware boosts/penalties, and rank match."""
    v = embed_query(q)
    D, I = index.search(v, k * 2)  # overfetch then rerank
    cand = []

    q_low = q.lower()

    for idx, score in zip(I[0], D[0]):
        c = CHUNKS[idx]
        text = c["text"]
        meta = c["meta"]
        t_low = text.lower()

        # ---- Priority boost from ingest (preferred), else filename heuristic
        prio = int(meta.get("priority", 0))
        if prio:
            priority_boost = {1: 0.0, 2: 0.20, 3: 0.40}.get(prio, 0.0)
        else:
            fname_heur = os.path.basename(meta.get("source", "")).lower()
            if "nttv rank requirements" in fname_heur:
                priority_boost = 0.40
            elif "nttv training reference" in fname_heur or "technique descriptions" in fname_heur:
                priority_boost = 0.20
            else:
                priority_boost = 0.0

        # ---- Generic keyword nudges (small)
        keyword_boost = 0.0
        if "ryu" in t_low or "ryū" in t_low:
            keyword_boost += 0.10
        if "school" in t_low or "schools" in t_low:
            keyword_boost += 0.05
        if "bujinkan" in t_low:
            keyword_boost += 0.05

        # ---- Query-aware boosts/penalties (STRONG for core concepts)
        qt_boost = 0.0

        # Kihon Happo
        if "kihon happo" in q_low and "kihon happo" in t_low:
            qt_boost += 0.60

        # Sanshin
        ask_sanshin = ("sanshin" in q_low) or ("san shin" in q_low)
        has_sanshin = ("sanshin" in t_low) or ("san shin" in t_low) or ("sanshin no kata" in t_low)
        if ask_sanshin and has_sanshin:
            qt_boost += 0.45

        # Kyusho
        if "kyusho" in q_low and "kyusho" in t_low:
            qt_boost += 0.25

        # Boshi/Shito names
        ask_boshi = ("boshi ken" in q_low) or ("shito ken" in q_low)
        has_boshi = ("boshi ken" in t_low) or ("shito ken" in t_low)
        if ask_boshi and has_boshi:
            qt_boost += 0.45

        # Weapons cues
        weapon_terms = [
            "hanbo","hanbō","rokushakubo","rokushaku","katana","tanto","shoto","shōtō",
            "kusari","fundo","kusari fundo","kyoketsu","shoge","shōge","shuko","shukō",
            "jutte","jitte","tessen","kunai","shuriken","senban","shaken"
        ]
        ask_weapon = (
            any(w in q_low for w in weapon_terms)
            or ("weapon" in q_low) or ("weapons" in q_low)
            or ("what rank" in q_low) or ("introduced at" in q_low)
            or ("when do i learn" in q_low)
        )
        has_weaponish = any(w in t_low for w in weapon_terms) or ("[weapon]" in t_low) or ("weapons reference" in t_low)
        if ask_weapon and has_weaponish:
            qt_boost += 0.55

        # Filename heuristic: prefer Weapons Reference / Glossary for weapons Qs
        fname = os.path.basename(meta.get("source") or "").lower()
        if ask_weapon and ("weapons reference" in fname or "glossary" in fname):
            qt_boost += 0.25

        # Schools / ryū boost
        school_aliases = []
        for canon, aliases in SCHOOL_ALIASES.items():
            school_aliases.extend([canon.lower()] + [a.lower() for a in aliases])
        if any(a in q_low for a in school_aliases) and any(a in t_low for a in school_aliases):
            qt_boost += 0.45

        # Leadership boost
        ask_soke = any(t in q_low for t in ["soke","sōke","grandmaster","headmaster","current head","current grandmaster"])
        has_soke = ("[sokeship]" in t_low) or (" soke" in t_low) or (" sōke" in t_low)
        if ask_soke and (has_soke or "leadership" in fname):
            qt_boost += 0.60
            if "leadership" in fname:
                qt_boost += 0.20

        # --- Technique name nudge (from Technique Descriptions)
        technique_terms = [
            "omote gyaku","ura gyaku","musha dori","take ori","hon gyaku jime","oni kudaki",
            "ude garame","ganseki otoshi","juji gatame","omoplata","te hodoki","tai hodoki",
        ]
        ask_tech = any(t in q_low for t in technique_terms) or ("what is" in q_low and len(q_low.split()) <= 6)
        has_tech = any(t in t_low for t in technique_terms) or ("technique descriptions" in fname)
        if ask_tech and has_tech:
            qt_boost += 0.55

        # --- Kata-specific boost so kata queries grab the technique table (PATCH)
        kata_boost = 0.0
        ask_kata = (" kata" in q_low) or ("no kata" in q_low) or (" kata?" in q_low)
        has_kata = (" kata" in t_low) or ("no kata" in t_low)
        if ask_kata and has_kata:
            kata_boost += 0.50

        # Offtopic penalties / lore / length
        offtopic_penalty = 0.0
        if "kihon happo" in q_low and "kyusho" in t_low: offtopic_penalty += 0.15
        if "kyusho" in q_low and "kihon happo" in t_low: offtopic_penalty += 0.15
        if ask_sanshin and "kyusho" in t_low: offtopic_penalty += 0.12

        lore_penalty = 0.0
        if any(k in t_low for k in ["sarutobi", "sasuke", "leaping from tree", "legend", "folklore"]):
            lore_penalty += 0.10

        length_penalty = min(len(text) / 2000.0, 0.3)

        # Exact rank match
        rank_boost = 0.0
        for rank in ["10th kyu","9th kyu","8th kyu","7th kyu","6th kyu","5th kyu","4th kyu","3rd kyu","2nd kyu","1st kyu"]:
            if rank in q_low and rank in t_low:
                rank_boost += 0.50

        new_score = (float(score)
                     + priority_boost
                     + keyword_boost
                     + qt_boost
                     + rank_boost
                     + kata_boost      # <--- include PATCH boost
                     - length_penalty
                     - offtopic_penalty
                     - lore_penalty)

        cand.append((new_score, {
            "text": text,
            "meta": meta,
            "source": meta.get("source"),
            "page": meta.get("page"),
            "score": float(score),
            "rerank_score": float(new_score),
        }))

    cand.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in cand[:k]]

def build_context(snippets: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    """Concatenate top-k snippets into a context block with a cap."""
    lines, total = [], 0
    for i, s in enumerate(snippets, 1):
        tag = f"[{i}] {os.path.basename(s['source'])}"
        if s.get("page"):
            tag += f" (p. {s['page']})"
        block = f"{tag}\n{s['text']}\n\n---\n"
        if total + len(block) > max_chars:
            break
        lines.append(block)
        total += len(block)
    return "".join(lines)

def retrieval_quality(hits: List[Dict[str, Any]]) -> float:
    if not hits:
        return 0.0
    return max(h.get("rerank_score", h.get("score", 0.0)) for h in hits)

# ---------------------------
# Injectors & helpers
# ---------------------------
def _gather_full_text_for_source(name_contains: str) -> Tuple[str, Optional[str]]:
    name_low = (name_contains or "").lower()
    parts, path = [], None
    for c in CHUNKS:
        src = (c["meta"].get("source") or "")
        if name_low in src.lower():
            parts.append(c["text"])
            path = src
    return ("\n\n".join(parts), path)

def inject_rank_passage_if_needed(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ql = question.lower()
    if not any(t in ql for t in ["kyu", "shodan", "rank requirement", "rank requirements"]):
        return hits
    txt, path = _gather_full_text_for_source("nttv rank requirements")
    if not txt:
        return hits
    synth = {
        "text": txt,
        "meta": {"priority": 1, "source": path or "nttv rank requirements.txt (synthetic)"},
        "source": path or "nttv rank requirements.txt (synthetic)",
        "page": None,
        "score": 1.0,
        "rerank_score": 997.0,
    }
    return [synth] + hits

def inject_leadership_passage_if_needed(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ql = question.lower()
    if not any(t in ql for t in ["soke","sōke","grandmaster","headmaster","current head","current grandmaster"]):
        return hits
    txt, path = _gather_full_text_for_source("bujinkan leadership and wisdom")
    if not txt:
        return hits
    synth = {
        "text": txt,
        "meta": {"priority": 1, "source": path or "Bujinkan Leadership and Wisdom.txt (synthetic)"},
        "source": path or "Bujinkan Leadership and Wisdom.txt (synthetic)",
        "page": None,
        "score": 1.0,
        "rerank_score": 998.0,
    }
    return [synth] + hits

def inject_schools_passage_if_needed(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ql = question.lower()
    if not any(t in ql for t in ["school", "schools", "ryu", "ryū", "bujinkan"]):
        return hits
    txt, path = _gather_full_text_for_source("schools of the bujinkan summaries")
    if not txt:
        return hits
    synth = {
        "text": txt,
        "meta": {"priority": 1, "source": path or "Schools of the Bujinkan Summaries.txt (synthetic)"},
        "source": path or "Schools of the Bujinkan Summaries.txt (synthetic)",
        "page": None,
        "score": 1.0,
        "rerank_score": 995.0,
    }
    return [synth] + hits

def inject_weapons_passage_if_needed(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepend the full NTTV Weapons Reference when the question mentions a weapon or 'rank/learn' for weapons."""
    ql = question.lower()
    weapon_triggers = [
        "hanbo","hanbō","rokushakubo","rokushaku","katana","tanto","shoto","shōtō",
        "kusari","fundo","kusari fundo","kyoketsu","shoge","shōge","shuko","shukō",
        "jutte","jitte","tessen","kunai","shuriken","senban","shaken","throwing star","throwing spike",
        "weapon","weapons","what rank","when do i learn","introduced at"
    ]
    if not any(t in ql for t in weapon_triggers):
        return hits
    txt, path = _gather_full_text_for_source("weapons reference")
    if not txt:
        return hits
    synth = {
        "text": txt,
        "meta": {"priority": 1, "source": path or "NTTV Weapons Reference.txt (synthetic)"},
        "source": path or "NTTV Weapons Reference.txt (synthetic)",
        "page": None,
        "score": 1.0,
        "rerank_score": 996.0,
    }
    return [synth] + hits

def inject_techniques_passage_if_needed(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepend the full Technique Descriptions when a technique-style question is asked (but NOT for concepts)."""
    ql = question.lower()

    # 🚫 Do NOT inject techniques for concept queries (these have their own extractors)
    if any(b in ql for b in ["kihon happo", "sanshin", "school", "schools", "ryu", "ryū"]):
        return hits

    triggers = [
        "what is", "define", "explain",
        "gyaku", "kudaki", "dori", "gatame", "ganseki", "nage", "otoshi",
        "wrist lock", "shoulder lock", "armbar",
        "te hodoki", "tai hodoki",
        " no kata",
    ]
    if not any(t in ql for t in triggers):
        return hits

    txt, path = _gather_full_text_for_source("technique descriptions")
    if not txt:
        return hits

    synth = {
        "text": txt,
        "meta": {"priority": 1, "source": path or "Technique Descriptions.md (synthetic)"},
        "source": path or "Technique Descriptions.md (synthetic)",
        "page": None,
        "score": 1.0,
        "rerank_score": 994.0,
    }
    return [synth] + hits

def inject_kihon_passage_if_needed(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """If the question is about Kihon Happo, synthesize a concise passage with the two subsets + items."""
    ql = question.lower()
    if "kihon happo" not in ql:
        return hits

    # harvest relevant lines from existing CHUNKS
    kosshi_lines, torite_lines, defs = [], [], []

    def push_lines_from(text: str):
        for raw in text.splitlines():
            ln = raw.strip()
            if not ln:
                continue
            low = ln.lower()
            if "kihon happo" in low and 10 < len(ln) < 220:
                defs.append(ln.rstrip(" ;,"))
            if "kosshi" in low and "sanpo" in low:
                # "Kosshi Kihon Sanpo: a, b, c" OR bullets/semicolon lists
                tail = ln.split(":", 1)[1].strip() if ":" in ln else ln
                parts = [p.strip(" -•\t") for p in re.split(r"[;,]", tail) if 2 <= len(p.strip()) <= 60]
                kosshi_lines.extend(parts)
            if "torite" in low and ("goho" in low or "gohō" in low):
                tail = ln.split(":", 1)[1].strip() if ":" in ln else ln
                parts = [p.strip(" -•\t") for p in re.split(r"[;,]", tail) if 2 <= len(p.strip()) <= 60]
                torite_lines.extend(parts)

    # scan top-N retrieved first, then a light scan across all CHUNKS if needed
    for p in hits[:8]:
        push_lines_from(p.get("text", ""))

    if (len(kosshi_lines) < 3 or len(torite_lines) < 5):
        for c in CHUNKS[:1000]:  # bounded scan
            src = (c["meta"].get("source") or "").lower()
            if not any(tag in src for tag in ["training reference", "rank requirements", "schools", "glossary", "technique descriptions"]):
                continue
            push_lines_from(c["text"])
            if len(kosshi_lines) >= 3 and len(torite_lines) >= 5 and defs:
                break

    # dedupe while preserving order
    def dedupe(seq):
        seen = set(); out = []
        for x in seq:
            if x and x not in seen:
                out.append(x); seen.add(x)
        return out

    kosshi = dedupe(kosshi_lines)[:3]
    torite = dedupe(torite_lines)[:5]

    if not (kosshi or torite or defs):
        return hits  # nothing to add

    # build the synthetic passage the kihon extractor loves
    parts = ["Kihon Happo consists of Kosshi Kihon Sanpo and Torite Goho."]
    if kosshi:
        parts.append("Kosshi Kihon Sanpo: " + ", ".join(kosshi) + ".")
    if torite:
        parts.append("Torite Goho: " + ", ".join(torite) + ".")
    if defs:
        parts.append(defs[0] if parts[-1].endswith(".") else (". " + defs[0]))

    body = " ".join(parts).strip()

    synth = {
        "text": body,
        "meta": {"priority": 1, "source": "Kihon Happo (synthetic)"},
        "source": "Kihon Happo (synthetic)",
        "page": None,
        "score": 1.0,
        "rerank_score": 998.0,
    }
    return [synth] + hits

# --- Single-technique CSV fast-path (parsing & render) ------------------------
def _parse_tech_csv_line(line: str) -> Optional[Dict[str, str]]:
    """
    Parse a single technique CSV row from Technique Descriptions.md.

    Expected logical columns (min 12):
      0 name
      1 japanese
      2 english
      3 family (e.g., 'Kihon Happo - Kosshi')
      4 rank_intro (e.g., '7th Kyu')
      5 approved (✅/False/True)
      6 focus
      7 safety
      8 partner_required (True/False)
      9 solo (True/False)
      10 tags (pipe-separated)
      11+ definition (may include commas)

    Returns a dict or None if we can't parse.
    """
    if not line or "," not in line:
        return None
    # split once for the first 11 fields; the rest join into definition
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 12:
        return None
    head = parts[:11]
    definition = ",".join(parts[11:]).strip()  # keep commas in definition

    name = head[0]
    japanese = head[1]
    english = head[2]
    family = head[3]
    rank_intro = head[4]
    approved = head[5]
    focus = head[6]
    safety = head[7]
    partner_required = head[8]
    solo = head[9]
    tags = head[10]

    return {
        "name": name,
        "japanese": japanese,
        "english": english,
        "family": family,
        "rank_intro": rank_intro,
        "approved": approved,
        "focus": focus,
        "safety": safety,
        "partner_required": partner_required,
        "solo": solo,
        "tags": tags,
        "definition": definition,
    }

def _render_single_technique(row: Dict[str, str], *, bullets: bool, tone: str, detail_mode: str) -> str:
    """
    Format a single technique into bullets or paragraph. 'detail_mode' in {"Brief","Standard","Full"}.
    """
    title  = row.get("name","Technique")
    jp     = row.get("japanese","")
    en     = row.get("english","")
    family = row.get("family","")
    rank   = row.get("rank_intro","")
    focus  = row.get("focus","")
    safety = row.get("safety","")
    part   = row.get("partner_required","")
    solo   = row.get("solo","")
    tags   = row.get("tags","")
    defin  = (row.get("definition") or "").strip()

    if detail_mode == "Brief":
        brief = [f"{title}:"]
        if en:  brief.append(f"- English: {en}")
        if jp:  brief.append(f"- Japanese: {jp}")
        if defin: brief.append(f"- Definition: {defin if defin.endswith('.') else defin + '.'}")
        body = "\n".join(brief)

    elif detail_mode == "Standard":
        std = [f"{title}:"]
        if en:      std.append(f"- English: {en}")
        if jp:      std.append(f"- Japanese: {jp}")
        if family:  std.append(f"- Family: {family}")
        if rank:    std.append(f"- Rank intro: {rank}")
        if focus:   std.append(f"- Focus: {focus}")
        if defin:   std.append(f"- Definition: {defin if defin.endswith('.') else defin + '.'}")
        body = "\n".join(std)

    else:  # Full
        full = [f"{title}:"]
        if jp:      full.append(f"- Japanese: {jp}")
        if en:      full.append(f"- English: {en}")
        if family:  full.append(f"- Family: {family}")
        if rank:    full.append(f"- Rank intro: {rank}")
        if focus:   full.append(f"- Focus: {focus}")
        if safety:  full.append(f"- Safety: {safety}")
        if part:    full.append(f"- Partner required: {part}")
        if solo:    full.append(f"- Solo: {solo}")
        if tags:    full.append(f"- Tags: {tags}")
        if defin:   full.append(f"- Definition: {defin if defin.endswith('.') else defin + '.'}")
        body = "\n".join(full)

    return _render_det(body, bullets=bullets, tone=tone)

def answer_single_technique_if_synthetic(passages: List[Dict[str, Any]], *, bullets: bool, tone: str, detail_mode: str) -> Optional[str]:
    """
    If the first passage is our synthetic single-technique CSV line, parse & render it now.
    """
    if not passages:
        return None
    top = passages[0]
    src = (top.get("source") or "").lower()
    if "technique descriptions (synthetic line)" not in src:
        return None
    line = (top.get("text") or "").strip()
    row = _parse_tech_csv_line(line)
    if not row:
        return None
    return _render_single_technique(row, bullets=bullets, tone=tone, detail_mode=detail_mode)


# --- Technique CSV line injector (for single-technique queries) ----------------
import unicodedata
import re as _re2  # avoid shadowing

def _fold(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()

def _is_single_technique_query(q: str) -> Optional[str]:
    """
    Return a candidate technique name if the query looks like a single technique ask,
    else None. Handles 'explain/define/what is ... (no kata)?'
    """
    ql = (q or "").strip().lower()
    # ignore concept/bucket queries
    for ban in ("kihon happo", "kihon happō", "sanshin", "school", "schools", "ryu", "ryū"):
        if ban in ql:
            return None
    m = _re2.search(r"(?:what\s+is|define|explain)\s+(.+)$", q, flags=_re2.I)
    cand = (m.group(1) if m else q).strip().rstrip("?!.")
    # trim generic suffixes
    cand = _re2.sub(r"\b(technique|in ninjutsu|in bujinkan)\b", "", cand, flags=_re2.I).strip()
    return cand if 2 <= len(cand) <= 80 else None

def _tech_name_variants(name: str) -> list[str]:
    v = [name.strip()]
    ln = name.strip().lower()
    # add/remove 'no kata'
    if ln.endswith(" no kata"):
        v.append(name[:-8].strip())
    else:
        v.append(f"{name} no Kata")
    # hyphen-insensitive
    nh = name.replace("-", " ")
    if nh != name:
        v.append(nh)
        if nh.lower().endswith(" no kata"):
            v.append(nh[:-8].strip())
        else:
            v.append(f"{nh} no Kata")
    # folded variant
    v.append(_fold(name))
    # de-dup preserve order
    seen = set(); out = []
    for x in v:
        if x and x not in seen:
            out.append(x); seen.add(x)
    return out

def _find_tech_line_in_chunks(name_variants: list[str]) -> Optional[str]:
    """
    Scan all CHUNKS for lines from Technique Descriptions.md whose first CSV cell
    (technique name) matches any variant (macron-insensitive).
    Return the full CSV line if found.
    """
    folded_targets = {_fold(v) for v in name_variants}
    for c in CHUNKS:
        src = (c["meta"].get("source") or "").lower()
        if "technique descriptions.md" not in src:
            continue
        for raw in c["text"].splitlines():
            line = raw.strip()
            if not line or "," not in line:
                continue
            first = line.split(",", 1)[0].strip()
            if _fold(first) in folded_targets:
                return line
    return None

def inject_specific_technique_line_if_needed(question: str, passages: list[dict]) -> list[dict]:
    cand = _is_single_technique_query(question)
    if not cand:
        return passages

    variants = _tech_name_variants(cand)
    line = _find_tech_line_in_chunks(variants)  # keep your existing scanner
    if not line:
        return passages

    synth = {
        "text": line,
        "meta": {"source": "Technique Descriptions (synthetic line)", "priority": 1},
        "source": "Technique Descriptions (synthetic line)",
        "page": None,
        "score": 1.0,
        "rerank_score": 1.0,
    }
    if not passages or passages[0].get("text") != line:
        return [synth] + passages
    return passages


# ---------------------------
# LLM backend (fallback)
# ---------------------------
def call_llm(prompt: str, system: str = "You are a precise assistant. Use only the provided context.") -> Tuple[str, str]:
    import requests
    model = os.environ.get("MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENROUTER_API_BASE") or os.environ.get("LM_STUDIO_BASE_URL") or "http://localhost:1234/v1"

    headers = {"Content-Type": "application/json"}
    if "openai" in base or "openrouter" in base:
        headers["Authorization"] = f"Bearer {api_key}" if api_key else ""

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 600,
    }

    try:
        r = requests.post(f"{base}/chat/completions", headers=headers, json=body, timeout=30)
        r.raise_for_status()
        data = r.json()
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        return text, json.dumps(data)[:4000]
    except Exception as e:
        return "", f'{{"error":"{type(e).__name__}","detail":"{str(e)}"}}'

# ---------------------------
# Prompt & answer planner
# ---------------------------
def build_prompt(context: str, question: str) -> str:
    return (
        "You must answer using ONLY the context below.\n"
        "Be concise but complete; avoid filler.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

# ---------------------------
# UI helpers for deterministic rendering
# ---------------------------
def _apply_tone(text: str, tone: str) -> str:
    """Light-touch tone adjustments; Chatty adds a friendly closer for non-bullets."""
    if tone == "Chatty":
        if "\n" not in text.strip():
            return text.strip() + " Want a quick drill or a bit of history too?"
        return text
    return text

def _bullets_to_paragraph(text: str) -> str:
    """Convert our fielded bullet output into a compact paragraph."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return text
    head = lines[0].rstrip(":")
    fields = {}
    for ln in lines[1:]:
        if ln.startswith("- "):
            ln = ln[2:]
        if ":" in ln:
            k, v = ln.split(":", 1)
            fields[k.strip().lower()] = v.strip().rstrip(".")
    segs = [head + ":"]
    if "translation" in fields:
        segs.append(f'“{fields["translation"]}”.')
    if "type" in fields:
        segs.append(f'Type: {fields["type"]}.')
    if "focus" in fields:
        segs.append(f'Focus: {fields["focus"]}.')
    if "weapons" in fields:
        segs.append(f'Weapons: {fields["weapons"]}.')
    if "notes" in fields:
        segs.append(f'Notes: {fields["notes"]}.')
    return " ".join(segs).strip()

def _render_det(text: str, *, bullets: bool, tone: str) -> str:
    """
    Deterministic renderer:
    - Bullets + Chatty: prepend a synthesized 'Quick take' line from the fields.
    - Bullets + Crisp: return bullets as-is.
    - Paragraph modes: convert bullets to paragraph then tone-adjust.
    """
    if bullets:
        if tone == "Chatty":
            lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
            title = lines[0].rstrip(":") if lines else ""
            fields = {}
            for ln in lines[1:]:
                if ln.startswith("- "):
                    ln = ln[2:]
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    fields[k.strip().lower()] = v.strip()
            trans = fields.get("translation")
            typ = fields.get("type")
            focus = fields.get("focus")
            quick_bits = []
            if trans: quick_bits.append(trans)
            if typ: quick_bits.append(typ)
            quick = " — ".join(quick_bits) if quick_bits else None
            if quick and focus:
                summary = f"Quick take: {quick}; focus on {focus}."
            elif quick:
                summary = f"Quick take: {quick}."
            elif focus:
                summary = f"Quick take: focus on {focus}."
            else:
                summary = None

            out = []
            if summary:
                out.append(summary)
            out.append(text.strip())
            out = "\n".join(out)
            if not out.strip().endswith("?"):
                out += "\n\nWant examples, drills, or lineage notes next?"
            return out
        else:
            return text.strip()

    para = _bullets_to_paragraph(text)
    return _apply_tone(para, tone)

# --- School intent detection ---
def is_school_query(question: str) -> bool:
    ql = question.lower()
    # any canonical/alias token OR generic '... ryu'
    for canon, aliases in SCHOOL_ALIASES.items():
        tokens = [canon.lower()] + [a.lower() for a in aliases]
        if any(tok in ql for tok in tokens):
            return True
    return (" ryu" in ql) or (" ryū" in ql)

def is_soke_query(q: str) -> bool:
    ql = q.lower()
    # cover accents and common phrasings
    return any(token in ql for token in [
        "soke", "sōke", "current soke", "who is the soke", "who is the sōke",
        "grandmaster", "current grandmaster", "who is the grandmaster"
    ])


# ---------------------------
# Core RAG pipeline
# ---------------------------
def answer_with_rag(question: str, k: int = TOP_K) -> Tuple[str, List[Dict[str, Any]], str]:
    # 1) Retrieve
    hits = retrieve(question, k=k)

    # 2) Inject domain-critical sources
    hits = inject_rank_passage_if_needed(question, hits)
    hits = inject_leadership_passage_if_needed(question, hits)
    hits = inject_schools_passage_if_needed(question, hits)
    hits = inject_weapons_passage_if_needed(question, hits)
    hits = inject_kihon_passage_if_needed(question, hits)
    hits = inject_techniques_passage_if_needed(question, hits)
    hits = inject_specific_technique_line_if_needed(question, hits)

    # Fast-path: if we injected a single technique CSV line, answer immediately
    fast = answer_single_technique_if_synthetic(
        hits,
        bullets=(output_style == "Bullets"),
        tone=tone_style,
        detail_mode=TECH_DETAIL_MODE,
    )
    if fast:
        return f"🔒 Strict (context-only, explain)\n\n{fast}", hits, '{"det_path":"technique/single"}'
    
    # --- Leadership (Sōke) gets priority over school profile if asked directly
    if is_soke_query(question):
        ans = try_leadership(question, hits)
        if ans:
            return ans, hits, {"det_path": "leadership/soke"}

    # 2.4) Schools LIST short-circuit
    if is_school_list_query(question):
        try:
            list_ans = try_answer_schools_list(
                question, hits, bullets=(output_style == "Bullets")
            )
        except Exception:
            list_ans = None
        if list_ans:
            rendered = _render_det(list_ans, bullets=(output_style == "Bullets"), tone=tone_style)
            return (
                f"🔒 Strict (context-only, explain)\n\n{rendered}",
                hits,
                '{"det_path":"schools/list"}'
            )

    # 2.5) School PROFILE short-circuit
    if is_school_query(question):
        try:
            school_fact = try_answer_school_profile(
                question, hits, bullets=(output_style == "Bullets")
            )
        except Exception:
            school_fact = None
        if school_fact:
            rendered = _render_det(school_fact, bullets=(output_style == "Bullets"), tone=tone_style)
            return (
                f"🔒 Strict (context-only, explain)\n\n{rendered}",
                hits,
                '{"det_path":"schools/profile"}'
            )

        # fallback LLM for schools
        ctx = build_context(hits)
        prompt = build_prompt(ctx, question)
        text, raw = call_llm(prompt)
        if not text.strip():
            return "🔒 Strict (context-only)\n\n❌ Model returned no text.", hits, raw or "{}"
        return f"🔒 Strict (context-only, explain)\n\n{text.strip()}", hits, raw or "{}"

    # 3) Leadership short-circuit
    asking_soke = any(t in question.lower() for t in ["soke","sōke","grandmaster","headmaster","current head","current grandmaster"])
    if asking_soke:
        try:
            fact = try_leadership(question, hits)
        except Exception:
            fact = None
        if fact:
            return f"🔒 Strict (context-only, explain)\n\n{fact}", hits, '{"det_path":"leadership/soke"}'

    # 4) Weapon rank short-circuit
    try:
        wr = try_answer_weapon_rank(question, hits)
    except Exception:
        wr = None
    if wr:
        return f"🔒 Strict (context-only)\n\n{wr}", hits, '{"det_path":"weapons/rank"}'

    # 5) Rank requirements short-circuit (ENTIRE block)
    try:
        rr = try_answer_rank_requirements(question, hits)
    except Exception:
        rr = None
    if rr:
        rendered = _render_det(rr, bullets=(output_style == "Bullets"), tone=tone_style)
        return f"🔒 Strict (context-only, explain)\n\n{rendered}", hits, '{"det_path":"rank/requirements"}'

    # 5.5) Kihon Happo hard short-circuit (MUST run BEFORE generic dispatcher)
    q_low = (question or "").lower()
    if "kihon happo" in q_low or "kihon happō" in q_low:
        kihon_ans = try_answer_kihon_happo(question, hits)
        if kihon_ans:
            rendered = _render_det(kihon_ans, bullets=(output_style == "Bullets"), tone=tone_style)
            return f"🔒 Strict (context-only, explain)\n\n{rendered}", hits, '{"det_path":"deterministic/kihon"}'

    # 6) Deterministic dispatcher (techniques, kyusho, kihon via dispatcher, sanshin, etc.)
    fact = try_extract_answer(question, hits)
    if fact:
        rendered = _render_det(fact, bullets=(output_style == "Bullets"), tone=tone_style)
        ql = question.lower()
        looks_like_kata = (" kata" in ql) or ("no kata" in ql) or re.search(r"\bexplain\s+.+\s+no\s+kata\b", ql)
        det_tag = '{"det_path":"technique/core"}' if looks_like_kata else '{"det_path":"deterministic/core"}'
        return f"🔒 Strict (context-only, explain)\n\n{rendered}", hits, det_tag

    # 7) LLM fallback with retrieved context
    ctx = build_context(hits)
    prompt = build_prompt(ctx, question)
    text, raw = call_llm(prompt)
    if not text.strip():
        return "🔒 Strict (context-only)\n\n❌ Model returned no text.", hits, raw or "{}"
    return f"🔒 Strict (context-only, explain)\n\n{text.strip()}", hits, raw or "{}"


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="NTTV Chatbot (Local RAG)", page_icon="🥋", layout="wide")

st.title("🥋 NTTV Chatbot (Local RAG)")
with st.sidebar:
    st.markdown("### Options")
    show_debug = st.checkbox("Show debugging", value=True)

    st.markdown("### Output")
    output_style = st.radio("Format", ["Bullets", "Paragraph"], index=0, help="Affects deterministic answers only.")
    tone_style = st.radio("Tone", ["Crisp", "Chatty"], index=0, help="Affects deterministic answers only.")
    st.caption("Deterministic answers = school profiles, rank requirements, weapon-rank facts, technique definitions, etc.")

    # --- Technique detail level toggle (NEW) ---
    TECH_DETAIL_MODE = st.selectbox(
        "Technique detail",
        options=["Brief", "Standard", "Full"],
        index=1,  # default Standard
        help="How much detail to show for single-technique answers."
    )

    st.markdown("---")
    st.markdown("**Backend**")
    st.caption("Configure MODEL and API base via env vars (OpenAI/OpenRouter/LM Studio).")

q = st.text_input("Ask a question:", value="", placeholder="e.g., what is omote gyaku")
go = st.button("Ask", type="primary")

if go and q.strip():
    with st.spinner("Thinking..."):
        ans, top_passages, raw_json = answer_with_rag(q.strip())

    st.markdown("### Answer")
    st.write(ans)

    if show_debug:
        st.markdown("### Retrieved sources")
        for i, h in enumerate(top_passages, 1):
            name = os.path.basename(h.get("source") or "")
            st.write(f"[{i}] {name} — score {h.get('score', 0):.3f} — priority {int(h.get('meta',{}).get('priority',0))}")
        st.markdown("### Show raw model response")
        st.code(raw_json, language="json")
else:
    st.info("Enter a question and click **Ask**.")
