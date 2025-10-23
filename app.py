# app.py — Streamlit RAG with priority-aware retrieval, strict prompt,
# now using modular extractors (Sanshin, Kihon Happo, Schools) + Rank injector
import os, pickle, json, re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 🔹 NEW: import the extractor router
from extractors import try_extract_answer

# ✅ Force CPU for embeddings (important on Windows/PyTorch)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

# ------------- Config / env -------------
load_dotenv()

BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234")
# Ensure BASE_URL ends with /v1 for OpenAI-compatible servers (LM Studio / OpenRouter)
if not BASE_URL.rstrip("/").endswith("/v1"):
    BASE_URL = BASE_URL.rstrip("/") + "/v1"

API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")  # LM Studio ignores key; OpenRouter needs a real key
MODEL   = os.getenv("MODEL_NAME", "google_gemma-3-1b-it")  # your current local model
TOP_K   = int(os.getenv("TOP_K", "6"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "160"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "index"
META_PATH = INDEX_DIR / "meta.pkl"
FAISS_PATH = INDEX_DIR / "faiss.index"
CFG_PATH = INDEX_DIR / "config.json"

# ------------- Load index + embedder -------------
if not (FAISS_PATH.exists() and META_PATH.exists() and CFG_PATH.exists()):
    raise SystemExit("Index not found. Run `python ingest.py` first to build index/.")

with open(CFG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)
EMBED_MODEL_NAME = cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")

EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

index = faiss.read_index(str(FAISS_PATH))
with open(META_PATH, "rb") as f:
    CHUNKS: List[Dict[str, Any]] = pickle.load(f)

# ------------- Embedding + retrieval -------------
def embed_query(q: str) -> np.ndarray:
    v = EMBED_MODEL.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    return v

def retrieve(q: str, k: int = TOP_K):
    """Search FAISS, then rerank with filename priority, query-aware boosts/penalties, and rank match."""
    v = embed_query(q)
    D, I = index.search(v, k*2)  # overfetch then rerank
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

def build_context(snippets, max_chars: int = 6000):
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

def retrieval_quality(hits):
    """Return best rerank score (or base score) as a quality signal."""
    if not hits:
        return 0.0
    return max(h.get("rerank_score", h.get("score", 0.0)) for h in hits)

# --- Helper: ensure Rank Requirements text is available to extractors ---
def _find_rank_file_text() -> tuple[str | None, str | None]:
    """
    Search common locations for the rank requirements file and return (text, path).
    Looks under ./data and project root, case-insensitive patterns.
    """
    candidates = []
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

def inject_rank_passage_if_needed(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    If the user asked a 'kyu' question and none of the top hits are the Rank Requirements,
    prepend a synthetic passage containing the full rank file so extractors can work.
    """
    if "kyu" not in question.lower():
        return hits

    has_rank = any("rank requirements" in (h.get("source") or "").lower() for h in hits)
    if has_rank:
        return hits

    txt, path = _find_rank_file_text()
    if not txt:
        return hits

    # Make a synthetic high-priority passage at the front
    synth = {
        "text": txt,
        "meta": {"priority": 1, "source": path},
        "source": path,
        "page": None,
        "score": 1.0,
        "rerank_score": 999.0,  # ensure it’s at the top for extractors
    }
    return [synth] + hits

# ------------- Prompting + LLM call -------------
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

def build_user_prompt(question, passages):
    # Use a bit more context for rank questions
    ctx_limit = 6 if "kyu" in question.lower() else 4
    ctx = "\n\n".join(p["text"] for p in passages[:ctx_limit])
    return (
        "Answer the question using ONLY the context.\n"
        "Return exactly one or two sentences, no bullets, no intro phrases.\n\n"
        f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx}\n\nANSWER:\n"
    )


def clean_answer(s: str) -> str:
    s = s.strip()
    # remove opening filler
    s = re.sub(r"^(here( is|'s) (the )?answer[:,]?\s*)", "", s, flags=re.I)
    s = re.sub(r"^(based on|according to) (the )?(provided )?context[:,]?\s*", "", s, flags=re.I)
    # remove bullets / bracket refs
    s = re.sub(r"(?m)^\s*(?:[-*]\s+|\[\d+\]\s*)", "", s)
    s = re.sub(r"\[(?:\d+|[A-Za-z])\]", "", s)
    # collapse whitespace
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def call_model_with_fallback(client: OpenAI, model: str, system: str, user: str, max_tokens=512, temperature=0.2):
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
                {"role":"system","content":sys_msg},
                {"role":"user","content":user_msg}
            ],
            temperature=0.0,
            max_tokens=120,
        )
        txt = r.choices[0].message.content.strip()
        return txt if txt else raw
    except Exception:
        return raw

def answer_with_rag(question: str):
    # 🔹 Overfetch more when the query mentions "kyu" (ranks)
    k = TOP_K
    if "kyu" in question.lower():
        k = max(TOP_K * 3, TOP_K + 6)

    # Retrieve
    hits = retrieve(question, k=k)

    # 🔹 Ensure rank file is present for kyu-style questions
    hits = inject_rank_passage_if_needed(question, hits)

    ctx = build_context(hits)
    best = retrieval_quality(hits)

    # ✅ Try deterministic extractors first (Sanshin, Kihon Happo, Schools, Rank-Striking, etc.)
    fact = try_extract_answer(question, hits)
    if fact:
        return f"🔒 Strict (context-only)\n\n{fact}", hits, "{}"

    # Decide strict vs hybrid
    weak_thresh_val = float(os.getenv("WEAK_THRESH", "0.35"))
    use_hybrid = best < weak_thresh_val
    system = HYBRID_SYSTEM if use_hybrid else STRICT_SYSTEM
    mode_note = "🧪 Hybrid (context + general knowledge)" if use_hybrid else "🔒 Strict (context-only)"

    # Build messages
    user = build_user_prompt(question, hits)

    # Call LLM
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    content, raw = call_model_with_fallback(
        client=client,
        model=MODEL,
        system=system,
        user=user,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    # 1) Clean the first draft
    content = clean_answer(content) if content else content

    # 2) Polish phrasing EXCEPT for rank questions
    if "kyu" not in question.lower():
        content = polish_answer(content, client, MODEL)

    # Return final
    return f"{mode_note}\n\n{content if content else '❌ Model returned no text.'}", hits, raw

# ------------- UI -------------
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
    show_debug = st.checkbox("Show debugging info (sources & raw model)", value=False)
    st.markdown("---")
    st.write("Tip: update your data in `/data`, run `python ingest.py`, then refresh.")

# Chat input
q = st.text_input("Ask a question:", placeholder="e.g., What is the weapon for 7th kyu?")
if st.button("Ask") or (q and st.session_state.get("auto_run", False)):
    if not q.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer, top_passages, raw_json = answer_with_rag(q)

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
