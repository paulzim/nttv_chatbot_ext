# app.py
import os
import json
import pickle
import time
from typing import List, Dict, Any, Optional, Tuple

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

# Deterministic extractors
from extractors import try_extract_answer
from extractors.schools import try_answer_school_profile
from extractors.leadership import try_extract_answer as try_leadership
from extractors.weapons import try_answer_weapon_rank
from extractors.rank import try_answer_rank_requirements

# ---------------------------
# Load index & metadata
# ---------------------------
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
CONFIG_PATH = os.path.join(INDEX_DIR, "config.json")
META_PATH = os.path.join(INDEX_DIR, "meta.pkl")

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Missing index config: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

EMBED_MODEL_NAME = cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
FAISS_PATH = cfg.get("faiss_path") or os.path.join(INDEX_DIR, "faiss.index")
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

        # Sanshin (catch "sanshin", "san shin", "sanshin no kata")
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

        # ---- Strong query-aware boost for weapons Qs when chunk looks like weapons content
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

        # Filename heuristic: prefer the Weapons Reference / Glossary for weapons questions
        fname = os.path.basename(meta.get("source", "")).lower()
        if ask_weapon and ("weapons reference" in fname or "glossary" in fname):
            qt_boost += 0.25

        # ---- Schools / ryū boost
        school_aliases = [
            "gyokko-ryu","gyokko ryu","gyokko-ryū","gyokko ryū",
            "koto-ryu","koto ryu","koto-ryū","koto ryū",
            "togakure-ryu","togakure ryu","togakure-ryū","togakure ryū",
            "shinden fudo-ryu","shinden fudo ryu","shinden fudō-ryū","shinden fudō ryū",
            "kukishinden-ryu","kukishinden ryu","kukishinden-ryū","kukishinden ryū",
            "takagi yoshin-ryu","takagi yoshin ryu","takagi yōshin-ryū","takagi yōshin ryū",
            "gikan-ryu","gikan ryu","gikan-ryū","gikan ryū",
            "gyokushin-ryu","gyokushin ryu","gyokushin-ryū","gyokushin ryū",
            "kumogakure-ryu","kumogakure ryu","kumogakure-ryū","kumogakure ryū",
        ]
        if any(a in q_low for a in school_aliases) and any(a in t_low for a in school_aliases):
            qt_boost += 0.45

        # ---- Leadership boost
        ask_soke = any(t in q_low for t in ["soke","sōke","grandmaster","headmaster","current head","current grandmaster"])
        has_soke = ("[sokeship]" in t_low) or (" soke" in t_low) or (" sōke" in t_low)
        if ask_soke and (has_soke or "leadership" in fname):
            qt_boost += 0.60
            if "leadership" in fname:
                qt_boost += 0.20

        # ---- Offtopic penalties / lore / length
        offtopic_penalty = 0.0
        if "kihon happo" in q_low and "kyusho" in t_low: offtopic_penalty += 0.15
        if "kyusho" in q_low and "kihon happo" in t_low: offtopic_penalty += 0.15
        if ask_sanshin and "kyusho" in t_low: offtopic_penalty += 0.12

        lore_penalty = 0.0
        if any(k in t_low for k in ["sarutobi", "sasuke", "leaping from tree", "legend", "folklore"]):
            lore_penalty += 0.10

        length_penalty = min(len(text) / 2000.0, 0.3)

        # ---- Exact rank match boost
        rank_boost = 0.0
        for rank in ["10th kyu","9th kyu","8th kyu","7th kyu","6th kyu","5th kyu","4th kyu","3rd kyu","2nd kyu","1st kyu"]:
            if rank in q_low and rank in t_low:
                rank_boost += 0.50

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

# ---------------------------
# LLM backend
# ---------------------------
def call_llm(prompt: str, system: str = "You are a precise assistant. Use only the provided context.") -> Tuple[str, str]:
    """
    Minimal OpenAI/OpenRouter/LM Studio fallback.
    Reads from environment:
      - OPENAI_API_KEY or OPENROUTER_API_KEY
      - OPENAI_BASE_URL / OPENROUTER_API_BASE or LM Studio at http://localhost:1234/v1
      - MODEL (e.g., openrouter/anthropic/sonnet, gpt-4o-mini, etc.)
    """
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
# Answer planner
# ---------------------------
def build_prompt(context: str, question: str) -> str:
    return (
        "You must answer using ONLY the context below.\n"
        "Be concise but complete; avoid filler.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

def answer_with_rag(question: str, k: int = TOP_K) -> Tuple[str, List[Dict[str, Any]], str]:
    # 1) Retrieve
    hits = retrieve(question, k=k)

    # 2) Inject domain-critical sources
    hits = inject_rank_passage_if_needed(question, hits)
    hits = inject_leadership_passage_if_needed(question, hits)
    hits = inject_schools_passage_if_needed(question, hits)
    hits = inject_weapons_passage_if_needed(question, hits)

    # 3) Leadership short-circuit
    asking_soke = any(t in question.lower() for t in ["soke","sōke","grandmaster","headmaster","current head","current grandmaster"])
    if asking_soke:
        fact = None
        try:
            fact = try_leadership(question, hits)
        except Exception:
            fact = None
        if fact:
            return f"🔒 Strict (context-only, explain)\n\n{fact}", hits, '{"det_path":"leadership/soke"}'

    # 4) Weapon rank short-circuit (single factual line)
    wr = None
    try:
        wr = try_answer_weapon_rank(question, hits)
    except Exception:
        wr = None
    if wr:
        return f"🔒 Strict (context-only)\n\n{wr}", hits, '{"det_path":"weapons/rank"}'

    # 5) Rank requirements (ENTIRE BLOCK) short-circuit
    rr = None
    try:
        rr = try_answer_rank_requirements(question, hits)
    except Exception:
        rr = None
    if rr:
        return f"🔒 Strict (context-only, explain)\n\n{rr}", hits, '{"det_path":"rank/requirements"}'

    # 6) Deterministic extractors (kyusho, kihon happo, sanshin, schools, etc.)
    fact = try_extract_answer(question, hits)

    # 7) School profile deterministic explainer
    school_fact = None
    try:
        school_fact = try_answer_school_profile(question, hits)
    except Exception:
        school_fact = None

    if school_fact:
        # Prefer the deterministic school profile when present
        return f"🔒 Strict (context-only, explain)\n\n{school_fact}", hits, '{"det_path":"schools/profile"}'

    if fact:
        # Use deterministic fact as final (already concise and grounded)
        return f"🔒 Strict (context-only, explain)\n\n{fact}", hits, '{"det_path":"deterministic/core"}'

    # 8) LLM fallback with retrieved context
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
    st.caption("Debugging shows retrieved sources and raw model response.")
    st.markdown("---")
    st.markdown("**Backend**")
    st.caption("Configure MODEL and API base via env vars (OpenAI/OpenRouter/LM Studio).")

q = st.text_input("Ask a question:", value="", placeholder="e.g., what are the rank requirements for 3rd kyu?")
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
