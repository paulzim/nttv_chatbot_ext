# app.py — Streamlit RAG with priority-aware retrieval and rank-file injection
import os, pickle, json, re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -------------------- Extractors (deterministic answers for key topics) --------------------
from extractors import try_extract_answer

# ✅ Force CPU for embeddings (important on Windows/PyTorch meta-tensor issues)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

# -------------------- Config / env --------------------
load_dotenv()

BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234")
# Ensure BASE_URL ends with /v1 for OpenAI-compatible servers (LM Studio / OpenRouter)
if not BASE_URL.rstrip("/").endswith("/v1"):
    BASE_URL = BASE_URL.rstrip("/") + "/v1"

API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")  # LM Studio ignores key; OpenRouter needs a real key
MODEL   = os.getenv("MODEL_NAME", "google_gemma-3-1b-it")
TOP_K   = int(os.getenv("TOP_K", "6"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "160"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "index"
META_PATH = INDEX_DIR / "meta.pkl"
FAISS_PATH = INDEX_DIR / "faiss.index"
CFG_PATH = INDEX_DIR / "config.json"

# -------------------- Load index + embedder --------------------
if not (FAISS_PATH.exists() and META_PATH.exists() and CFG_PATH.exists()):
    raise SystemExit("Index not found. Run `python ingest.py` first to build index/.")

with open(CFG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)
EMBED_MODEL_NAME = cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")

EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

index = faiss.read_index(str(FAISS_PATH))
with open(META_PATH, "rb") as f:
    CHUNKS: List[Dict[str, Any]] = pickle.load(f)

# -------------------- Embedding + retrieval --------------------
def embed_query(q: str) -> np.ndarray:
    v = EMBED_MODEL.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    return v

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
            fname = os.path.basename(meta.get("source", "")).lower()
            if "nttv rank requirements" in fname:
                priority_boost = 0.40
            elif "nttv training reference" in fname or "technique descriptions" in fname:
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

        # Sanshin (catch "sanshin", "san shin", "sanshin no kata")
        ask_sanshin = ("sanshin" in q_low) or ("san shin" in q_low)
        has_sanshin = ("sanshin" in t_low) or ("san shin" in t_low) or ("sanshin no kata" in t_low)
        if ask_sanshin and has_sanshin:
            qt_boost += 0.45

        # Kyusho
        if "kyusho" in q_low and "kyusho" in t_low:
            qt_boost += 0.25

        # Strike names (Boshi/Shito)
        ask_boshi = ("boshi ken" in q_low) or ("shito ken" in q_low)
        has_boshi = ("boshi ken" in t_low) or ("shito ken" in t_low)
        if ask_boshi and has_boshi:
            qt_boost += 0.45

        # NEW: Schools / ryū explicit boost (paired with "Bujinkan")
        ask_schools = (("school" in q_low) or ("schools" in q_low) or ("ryu" in q_low) or ("ryū" in q_low)) and ("bujinkan" in q_low)
        has_schools = (("school" in t_low) or ("schools" in t_low) or ("ryu" in t_low) or ("ryū" in t_low)) and ("bujinkan" in t_low)
        if ask_schools and has_schools:
            qt_boost += 0.55

        offtopic_penalty = 0.0
        if "kihon happo" in q_low and "kyusho" in t_low:
            offtopic_penalty += 0.15
        if "kyusho" in q_low and "kihon happo" in t_low:
            offtopic_penalty += 0.15
        if ask_sanshin and "kyusho" in t_low:
            offtopic_penalty += 0.12

        # ---- De-emphasize lore/anecdotes in strict facts mode
        lore_penalty = 0.0
        if any(k in t_low for k in ["sarutobi", "sasuke", "leaping from tree", "legend", "folklore"]):
            lore_penalty += 0.10

        # ---- Prefer concise chunks
        length_penalty = min(len(text) / 2000.0, 0.3)

        # ---- Exact rank match boost
        rank_boost = 0.0
        for rank in ["10th kyu", "9th kyu", "8th kyu", "7th kyu", "6th kyu",
                     "5th kyu", "4th kyu", "3rd kyu", "2nd kyu", "1st kyu"]:
            if rank in q_low and rank in t_low:
                rank_boost += 0.50  # strong boost for exact rank match

        # ---- Final rerank score
        new_score = (float(score)
                     + priority_boost
                     + keyword_boost
                     + qt_boost
                     + rank_boost
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
    """Return best rerank score (or base score) as a quality signal."""
    if not hits:
        return 0.0
    return max(h.get("rerank_score", h.get("score", 0.0)) for h in hits)

# -------------------- Rank-file injectors --------------------
def _find_rank_file_text() -> tuple[str | None, str | None]:
    """
    Filesystem fallback: search common locations for the rank requirements file and return (text, path).
    Looks under ./data and project root, case-insensitive patterns.
    """
    root = ROOT
    data_dir = root / "data"
    patterns = [
        "nttv rank requirements.txt",
        "rank requirements.txt",
        "*rank*requirements*.txt",
    ]
    search_dirs = [data_dir, root]
    seen = set()
    for d in search_dirs:
        if not d.exists():
            continue
        for pat in patterns:
            for p in d.glob(pat):
                lp = str(p).lower()
                if lp in seen:
                    continue
                seen.add(lp)
                try:
                    txt = p.read_text(encoding="utf-8", errors="replace")
                    if txt and "kyu" in txt.lower():
                        return txt, str(p)
                except Exception:
                    pass
    return None, None

def _gather_full_text_for_file(name_contains: str) -> tuple[str | None, str | None]:
    """
    Reconstruct full file text by concatenating all CHUNKS whose basename
    contains `name_contains` (case-insensitive). Returns (text, example_path).
    """
    want = name_contains.lower()
    matched = [c for c in CHUNKS if want in os.path.basename((c.get("meta", {}) or {}).get("source", "")).lower()]
    if not matched:
        return None, None
    parts = [c["text"] for c in matched if c.get("text")]
    full = "\n\n".join(parts).strip() if parts else None
    any_path = matched[0].get("meta", {}).get("source")
    return (full if full else None), any_path

def inject_rank_passage_if_needed(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For any 'kyu' question, prepend a synthetic passage containing the full
    Rank Requirements text so extractors always see the whole file.
    """
    if "kyu" not in question.lower():
        return hits

    # Prefer gathering from loaded CHUNKS (index truth)
    txt, path = _gather_full_text_for_file("nttv rank requirements")
    if not txt:
        # Fallback to filesystem scan
        txt, path = _find_rank_file_text()

    if not txt:
        return hits

    synth = {
        "text": txt,
        "meta": {"priority": 1, "source": path or "NTTV Rank Requirements (synthetic)"},
        "source": path or "NTTV Rank Requirements (synthetic)",
        "page": None,
        "score": 1.0,
        "rerank_score": 999.0,  # force top position
    }
    return [synth] + hits

# -------------------- Prompting + LLM call --------------------
STRICT_SYSTEM = (
    "You are the NTTV assistant. Answer ONLY from the provided context.\n"
    "Style: one or two short sentences, declarative, no preambles.\n"
    "Do not include citations, brackets, or mention files/sections.\n"
    "Use domain terms exactly as written in the context."
)

HYBRID_SYSTEM = (
    "You are the NTTV assistant. Prefer the provided context; if it is weak, you may fill small gaps "
    "with common knowledge.\n"
    "Style: one or two short sentences, declarative, no preambles.\n"
    "Do not include citations, brackets, or mention files/sections.\n"
    "Use domain terms exactly as written."
)

def build_user_prompt(question: str, passages: List[Dict[str, Any]]) -> str:
    # Use a bit more context for rank questions
    ctx_limit = 6 if "kyu" in question.lower() else 4
    ctx = "\n\n".join(p["text"] for p in passages[:ctx_limit])
    return (
        "Answer the question using ONLY the context.\n"
        "Return exactly one or two sentences, no bullets, no intro phrases.\n\n"
        f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx}\n\nANSWER:\n"
    )

def build_explanation_prompt(question: str, passages: List[Dict[str, Any]], fact_sentence: str) -> str:
    """
    Build a strict, context-only prompt that produces a short explanation.
    The first sentence must be the deterministic fact we already extracted.
    """
    # Use a bit more context for explanations
    ctx_limit = 8 if any(t in question.lower() for t in ["kyu", "kihon", "happo", "sanshin", "school", "ryu"]) else 6
    ctx = "\n\n".join(p["text"] for p in passages[:ctx_limit])
    return (
        "Using ONLY the provided context, write 2–4 short declarative sentences that explain the answer.\n"
        "Begin with this exact sentence (do not modify it):\n"
        f"{fact_sentence}\n\n"
        "Then add 1–3 sentences of rationale drawn from the context. "
        "No citations, no file names, no brackets.\n\n"
        f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx}\n\nEXPLANATION:\n"
    )

def clean_answer(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^(here( is|'s) (the )?answer[:,]?\s*)", "", s, flags=re.I)
    s = re.sub(r"^(based on|according to) (the )?(provided )?context[:,]?\s*", "", s, flags=re.I)
    s = re.sub(r"(?m)^\s*(?:[-*]\s+|\[\d+\]\s*)", "", s)
    s = re.sub(r"\[(?:\d+|[A-Za-z])\]", "", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def call_model_with_fallback(client: OpenAI, model: str, system: str, user: str,
                             max_tokens: int = 512, temperature: float = 0.2) -> tuple[str, str]:
    """
    Try /v1/chat/completions first, then fall back to /v1/completions.
    Return (content, raw_json_str) for debugging UI.
    """
    import json as _json

    # 1) chat.completions
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = None
        if getattr(r, "choices", None):
            ch0 = r.choices[0]
            if getattr(ch0, "message", None) and getattr(ch0.message, "content", None):
                content = ch0.message.content
            if not content and getattr(ch0, "text", None):
                content = ch0.text
        raw = r.model_dump_json() if hasattr(r, "model_dump_json") else _json.dumps(r, default=str)
        if content and content.strip():
            return content, raw
    except Exception as e:
        raw = f"chat.completions error: {e}"

    # 2) completions (fallback)
    prompt = f"{system}\n\n{user}"
    try:
        r2 = client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = None
        if getattr(r2, "choices", None):
            ch0 = r2.choices[0]
            if getattr(ch0, "text", None):
                content = ch0.text
        raw2 = r2.model_dump_json() if hasattr(r2, "model_dump_json") else _json.dumps(r2, default=str)
        if content and content.strip():
            return content, raw2
        return "", raw2
    except Exception as e:
        return "", f"{raw}\n\ncompletions error: {e}"

def polish_answer(raw: str, client: OpenAI, model: str) -> str:
    if not raw or len(raw) < 5:
        return raw
    sys_msg = (
        "Rewrite the user's draft into 1–2 short declarative sentences.\n"
        "Do not add new facts. Do not include citations or preambles."
    )
    user_msg = f"Draft:\n{raw}\n\nRewrite:"
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=120,
        )
        txt = r.choices[0].message.content.strip()
        return txt if txt else raw
    except Exception:
        return raw

# -------------------- Explanation helpers --------------------
def enrich_context_for_explanation(question: str, hits: list[dict], k_extra: int = 12) -> list[dict]:
    """
    For explanation mode only: if the question targets a known concept,
    run one extra targeted retrieval to enrich passages before prompting.
    """
    ql = question.lower()
    alt_q = None
    if "kihon" in ql or "happo" in ql or "happō" in ql:
        alt_q = "Kihon Happo Kosshi Kihon Sanpo Torite Goho definition list names"
    elif "sanshin" in ql or "san shin" in ql:
        alt_q = "Sanshin no Kata five forms list definition"
    elif "school" in ql or "schools" in ql or "ryu" in ql or "ryū" in ql:
        alt_q = "Bujinkan schools list ryu names summary"

    if not alt_q:
        return hits

    extra = retrieve(alt_q, k=k_extra)
    # Merge (extra first), then dedupe by (source, first 300 chars hash)
    merged = extra + hits
    seen = set()
    deduped = []
    for p in merged:
        key = (p.get("source"), hash(p.get("text", "")[:300]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped

def _ensure_sentence(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    return s if s.endswith((".", "!", "?")) else s + "."

def try_build_kihon_explanation(question: str, passages: List[Dict[str, Any]], fact_sentence: str) -> str | None:
    """
    Deterministically build a 2-sentence explanation for Kihon Happo:
      1) Keep the fact sentence as-is
      2) Append subset contents if we can parse them from context.
    """
    ql = question.lower()
    if not ("kihon" in ql or "happo" in ql or "happō" in ql):
        return None

    # tolerant patterns
    KOSHI_PAT = re.compile(r"(?i)\bkosshi?\s+kihon\s+sanpo\b|\bkoshi\s+sanpo\b")
    TORITE_PAT = re.compile(r"(?i)\btorite\s+goho(?:\s+gata)?\b")
    BULLET = re.compile(r"^\s*[•\-\u2022]\s*(.+?)\s*$")
    SEP = re.compile(r"[;,]")

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip())

    def _split_items(line: str) -> List[str]:
        parts = [p.strip(" -•\t") for p in SEP.split(line) if p.strip()]
        return [p for p in parts if 2 <= len(p) <= 60]

    def _collect_subset(lines: List[str], start_idx: int) -> List[str]:
        items = []
        i = start_idx + 1
        while i < len(lines):
            ln = lines[i]
            if not ln.strip():
                break
            if KOSHI_PAT.search(ln) or TORITE_PAT.search(ln):
                break
            m = BULLET.match(ln)
            if m:
                items.append(_norm(m.group(1)))
            elif SEP.search(ln) and len(ln) < 240:
                items.extend(_split_items(ln))
            i += 1
        # de-dup, preserve order
        seen, out = set(), []
        for x in items:
            k = _norm(x).lower()
            if k in seen:
                continue
            seen.add(k); out.append(_norm(x))
        return out

    koshi, torite = [], []

    for p in passages[:12]:
        text = p.get("text", "")
        if not text or len(text) < 20:
            continue
        lines = text.splitlines()
        for i, ln in enumerate(lines):
            if KOSHI_PAT.search(ln):
                koshi.extend(_collect_subset(lines, i))
            if TORITE_PAT.search(ln):
                torite.extend(_collect_subset(lines, i))

    # finalize unique lists
    def _dedupe(lst):
        seen, out = set(), []
        for x in lst:
            k = _norm(x).lower()
            if k in seen:
                continue
            seen.add(k); out.append(_norm(x))
        return out

    koshi = _dedupe(koshi)[:3]
    torite = _dedupe(torite)[:5]

    if not koshi and not torite:
        return None

    parts = [_ensure_sentence(fact_sentence)]
    if koshi:
        parts.append(f"Kosshi Kihon Sanpo: {', '.join(koshi)}.")
    if torite:
        parts.append(f"Torite Goho: {', '.join(torite)}.")
    return " ".join(parts)

def try_build_sanshin_explanation(question: str, passages: List[Dict[str, Any]]) -> str | None:
    ql = question.lower()
    if not ("sanshin" in ql or "san shin" in ql):
        return None

    SEP = re.compile(r"[;,]")
    def _norm(s: str): return re.sub(r"\s+", " ", s.strip())

    items = []
    for p in passages[:12]:
        t = p.get("text","")
        if not t:
            continue
        for line in t.splitlines():
            line = _norm(line)
            if len(line) > 240:
                continue
            # lines that mention sanshin and look like a list
            if ("sanshin" in line.lower()) and (SEP.search(line) or line.count(",") >= 2):
                parts = [x.strip(" -•\t") for x in re.split(r"[;,]", line) if x.strip()]
                items.extend(parts)
            # listy lines (fallback)
            elif SEP.search(line) and line.count(",") >= 2:
                items.extend([x.strip(" -•\t") for x in re.split(r"[;,]", line) if x.strip()])

    # dedupe, keep 5 if possible
    seen, dedup = set(), []
    for x in items:
        k = _norm(x).lower()
        if k in seen:
            continue
        seen.add(k); dedup.append(_norm(x))
    if len(dedup) >= 3:
        forms = ", ".join(dedup[:5])
        return f"Sanshin no Kata consists of five fundamental forms used to build structure, distance, and timing. Forms include: {forms}."
    return None

def try_build_schools_explanation(question: str, passages: List[Dict[str, Any]]) -> str | None:
    """
    Deterministic one-liner that lists Bujinkan schools when asked about 'schools' or 'ryu'.
    """
    ql = question.lower()
    if not ("school" in ql or "schools" in ql or "ryu" in ql or "ryū" in ql):
        return None

    names = []
    for p in passages[:12]:
        t = (p.get("text", "") or "").strip()
        if not t or "bujinkan" not in t.lower():
            continue
        for line in t.splitlines():
            L = line.strip()
            if len(L) < 20 or len(L) > 400:
                continue
            # harvest comma/semicolon lists that look like multiple proper nouns
            if ("," in L or ";" in L) and sum(ch in L for ch in ",;") >= 4:
                parts = [x.strip(" -•\t") for x in re.split(r"[;,]", L) if x.strip()]
                # keep name-like tokens (2–60 chars, avoid sentences)
                for p2 in parts:
                    if 2 <= len(p2) <= 60 and not p2.endswith((".", ":", ";")):
                        names.append(p2)

    # dedupe and cap
    seen, out = set(), []
    for x in names:
        k = x.strip().lower()
        if k in seen:
            continue
        seen.add(k); out.append(x.strip())

    if len(out) >= 5:
        return f"The Bujinkan encompasses classical lineages including: {', '.join(out[:9])}."
    return None

def try_build_strike_explanation(question: str, passages: list[dict]) -> str | None:
    """
    Deterministic explainer for specific strikes like Boshi Ken / Shito Ken.
    Returns the first clean definition line found in context.
    """
    ql = question.lower()
    strike_terms = ["boshi ken", "shito ken"]

    if not any(term in ql for term in strike_terms):
        return None

    for p in passages:
        txt = (p.get("text") or "").strip()
        if not txt:
            continue
        for line in txt.splitlines():
            l_low = line.lower()
            if any(term in l_low for term in strike_terms):
                clean = line.strip(" -•\t")
                if len(clean.split()) <= 3:
                    continue
                if not clean.endswith("."):
                    clean += "."
                return clean
    return None

# -------------------- RAG pipeline --------------------
def answer_with_rag(question: str, explain: bool = False):
    # Overfetch more when the query mentions "kyu" (ranks)
    k = TOP_K
    if "kyu" in question.lower():
        k = max(TOP_K * 4, TOP_K + 10)

    hits = retrieve(question, k=k)
    hits = inject_rank_passage_if_needed(question, hits)
    ctx = build_context(hits)
    best = retrieval_quality(hits)

    # Deterministic extractor path (rank/kihon/… if present)
    fact = try_extract_answer(question, hits)

    # NEW: Deterministic concept fallbacks even when explain=False
    if not fact:
        # Try to synthesize a strict, context-only answer for common concept queries
        det = try_build_schools_explanation(question, hits)
        if det:
            return f"🔒 Strict (context-only)\n\n{det}", hits, "{}"
        det = try_build_sanshin_explanation(question, hits)
        if det:
            return f"🔒 Strict (context-only)\n\n{det}", hits, "{}"
        det = try_build_strike_explanation(question, hits)
        if det:
            return f"🔒 Strict (context-only)\n\n{det}", hits, "{}"
        # Optional: Kihon Happo without explanation mode (use a safe fixed fact line)
        if ("kihon" in question.lower() or "happo" in question.lower() or "happō" in question.lower()):
            base_fact = "Kihon Happo consists of Kosshi Kihon Sanpo and Torite Goho."
            det = try_build_kihon_explanation(question, hits, base_fact)
            if det:
                return f"🔒 Strict (context-only)\n\n{det}", hits, "{}"

    if fact and not explain:
        return f"🔒 Strict (context-only)\n\n{fact}", hits, "{}"

    if fact and explain:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

        # Enrich context so the model (or deterministic explainer) has material
        explain_hits = enrich_context_for_explanation(question, hits, k_extra=max(TOP_K * 2, 12))

        # 🔒 Deterministic explanation attempts (no model)
        det_expl = try_build_kihon_explanation(question, explain_hits, fact)
        if det_expl:
            return f"🔒 Strict (context-only, explain)\n\n{det_expl}", explain_hits, "{}"

        det_expl = try_build_sanshin_explanation(question, explain_hits)
        if det_expl:
            return f"🔒 Strict (context-only, explain)\n\n{det_expl}", explain_hits, "{}"

        det_expl = try_build_schools_explanation(question, explain_hits)
        if det_expl:
            return f"🔒 Strict (context-only, explain)\n\n{det_expl}", explain_hits, "{}"

        det_expl = try_build_strike_explanation(question, explain_hits)
        if det_expl:
            return f"🔒 Strict (context-only, explain)\n\n{det_expl}", explain_hits, "{}"

        # Otherwise, ask the model to elaborate strictly from context
        user = build_explanation_prompt(question, explain_hits, fact)
        content, raw = call_model_with_fallback(
            client=client,
            model=MODEL,
            system=STRICT_SYSTEM,
            user=user,
            max_tokens=min(MAX_TOKENS, 240),
            temperature=0.0,
        )
        content = clean_answer(content) if content else content
        if not (content or "").strip().lower().startswith(fact.strip().lower()[:20]):
            content = f"{fact} " + (content or "")
        return f"🔒 Strict (context-only, explain)\n\n{content if content else fact}", explain_hits, raw

    # Strict vs hybrid selection
    weak_thresh_val = float(os.getenv("WEAK_THRESH", "0.35"))
    use_hybrid = best < weak_thresh_val
    system = HYBRID_SYSTEM if use_hybrid else STRICT_SYSTEM
    mode_note = "🧪 Hybrid (context + general knowledge)" if use_hybrid else "🔒 Strict (context-only)"

    user = build_user_prompt(question, hits)
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    content, raw = call_model_with_fallback(
        client=client,
        model=MODEL,
        system=system,
        user=user,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    content = clean_answer(content) if content else content
    if "kyu" not in question.lower():
        content = polish_answer(content, client, MODEL)
    return f"{mode_note}\n\n{content if content else '❌ Model returned no text.'}", hits, raw

# -------------------- UI --------------------
st.set_page_config(page_title="NTTV Chat", page_icon="💬")
st.title("💬 NTTV Chat (Local RAG)")
st.caption("Local retrieval with strict, context-first answers. Toggle debug to inspect sources.")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("Status")
    st.write(f"Model: `{MODEL}`")
    st.write(f"Server: `{BASE_URL}`")
    st.write(f"Top K: {TOP_K}  |  Max tokens: {MAX_TOKENS}")
    weak_thresh_val = float(os.getenv("WEAK_THRESH", "0.35"))
    st.write(f"Weak retrieval threshold: {weak_thresh_val:.2f}")
    st.markdown("---")
    explain = st.toggle("Explanation mode", value=False, help="Add a brief, context-only rationale after the short fact.")
    show_debug = st.checkbox("Show debugging info (sources & raw model)", value=False)
    st.markdown("---")
    st.write("Tip: update your data in `/data`, run `python ingest.py`, then refresh.")

q = st.text_input("Ask a question:", placeholder="e.g., What are the schools of the Bujinkan?")
if st.button("Ask") or (q and st.session_state.get("auto_run", False)):
    if not q.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer, top_passages, raw_json = answer_with_rag(q, explain=explain)

        st.markdown("### Answer")
        st.write(answer)

        if show_debug:
            st.markdown("### Retrieved sources")
            for i, p in enumerate(top_passages, 1):
                fname = os.path.basename(p.get("source", "") or "")
                base = f"[{i}] {fname}"
                score = p.get("rerank_score", p.get("score", 0.0))
                prio = p.get("meta", {}).get("priority", None)
                suffix = f" — score {score:.3f}"
                if prio:
                    suffix += f" — priority {prio}"
                if p.get("page"):
                    suffix += f" — p.{p['page']}"
                st.write(base + suffix)
            with st.expander("Show raw model response"):
                st.code(raw_json, language="json")
