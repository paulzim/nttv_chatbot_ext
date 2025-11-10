# app.py — NTTV Chat (Always-Explain, Context-Only) with leadership-first + school concrete answers
import os, pickle, json, re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from extractors import try_extract_answer
from extractors.leadership import try_extract_answer as try_leadership
from extractors.schools import try_answer_school_profile  # NEW

# CPU embeddings
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

load_dotenv()

BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234")
if not BASE_URL.rstrip("/").endswith("/v1"):
    BASE_URL = BASE_URL.rstrip("/") + "/v1"

API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
MODEL   = os.getenv("MODEL_NAME", "google_gemma-3-1b-it")
TOP_K   = int(os.getenv("TOP_K", "6"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "240"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "index"
META_PATH = INDEX_DIR / "meta.pkl"
FAISS_PATH = INDEX_DIR / "faiss.index"
CFG_PATH = INDEX_DIR / "config.json"

if not (FAISS_PATH.exists() and META_PATH.exists() and CFG_PATH.exists()):
    raise SystemExit("Index not found. Run `python ingest.py` first to build index/.")

with open(CFG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)
EMBED_MODEL_NAME = cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

index = faiss.read_index(str(FAISS_PATH))
with open(META_PATH, "rb") as f:
    CHUNKS: List[Dict[str, Any]] = pickle.load(f)

# -------------------- Retrieval --------------------
def embed_query(q: str) -> np.ndarray:
    v = EMBED_MODEL.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    return v

def retrieve(q: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    v = embed_query(q)
    D, I = index.search(v, k * 2)
    cand = []

    q_low = q.lower().replace("gyokku ryu", "gyokko ryu").replace("gyokku-ryu", "gyokko-ryu")

    for idx, score in zip(I[0], D[0]):
        c = CHUNKS[idx]
        text = c["text"]
        meta = c["meta"]
        t_low = text.lower()

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

        keyword_boost = 0.0
        if "ryu" in t_low or "ryū" in t_low: keyword_boost += 0.10
        if "school" in t_low or "schools" in t_low: keyword_boost += 0.05
        if "bujinkan" in t_low: keyword_boost += 0.05

        qt_boost = 0.0
        if "kihon happo" in q_low and "kihon happo" in t_low: qt_boost += 0.60
        ask_sanshin = ("sanshin" in q_low) or ("san shin" in q_low)
        has_sanshin = ("sanshin" in t_low) or ("san shin" in t_low) or ("sanshin no kata" in t_low)
        if ask_sanshin and has_sanshin: qt_boost += 0.45
        if "kyusho" in q_low and "kyusho" in t_low: qt_boost += 0.25
        ask_boshi = ("boshi ken" in q_low) or ("shito ken" in q_low)
        has_boshi = ("boshi ken" in t_low) or ("shito ken" in t_low)
        if ask_boshi and has_boshi: qt_boost += 0.45

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

        ask_soke = any(t in q_low for t in ["soke","sōke","grandmaster","headmaster","current head","current grandmaster"])
        has_soke = ("[sokeship]" in t_low) or (" soke" in t_low) or (" sōke" in t_low)
        fname = os.path.basename(meta.get("source", "")).lower()
        if ask_soke and (has_soke or "leadership" in fname):
            qt_boost += 0.60
            if "leadership" in fname: qt_boost += 0.20

        weapon_terms = [
            "hanbo","hanbō","rokushakubo","rokushaku","katana","tanto","shoto",
            "kusari fundo","naginata","kyoketsu shoge","shuko","jutte","tessen",
            "kunai","shuriken","senban","shaken"
        ]
        ask_weapon = any(w in q_low for w in weapon_terms) or ("weapon" in q_low) or ("weapons" in q_low)
        has_weapon = any(w in t_low for w in weapon_terms) or ("[weapon]" in t_low) or ("weapons reference" in t_low)
        if ask_weapon and has_weapon: qt_boost += 0.30

        offtopic_penalty = 0.0
        if "kihon happo" in q_low and "kyusho" in t_low: offtopic_penalty += 0.15
        if "kyusho" in q_low and "kihon happo" in t_low: offtopic_penalty += 0.15
        if ask_sanshin and "kyusho" in t_low: offtopic_penalty += 0.12

        lore_penalty = 0.0
        if any(k in t_low for k in ["sarutobi", "sasuke", "leaping from tree", "legend", "folklore"]):
            lore_penalty += 0.10

        length_penalty = min(len(text) / 2000.0, 0.3)

        rank_boost = 0.0
        for rank in ["10th kyu","9th kyu","8th kyu","7th kyu","6th kyu","5th kyu","4th kyu","3rd kyu","2nd kyu","1st kyu"]:
            if rank in q_low and rank in t_low: rank_boost += 0.50

        new_score = (float(score) + priority_boost + keyword_boost + qt_boost + rank_boost
                     - length_penalty - offtopic_penalty - lore_penalty)

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

def build_context(snippets: List[Dict[str, Any]], max_chars: int = 6500) -> str:
    lines, total = [], 0
    for i, s in enumerate(snippets, 1):
        tag = f"[{i}] {os.path.basename(s['source'])}"
        if s.get("page"): tag += f" (p. {s['page']})"
        block = f"{tag}\n{s['text']}\n\n---\n"
        if total + len(block) > max_chars: break
        lines.append(block); total += len(block)
    return "".join(lines)

def retrieval_quality(hits: List[Dict[str, Any]]) -> float:
    if not hits: return 0.0
    return max(h.get("rerank_score", h.get("score", 0.0)) for h in hits)

# -------------------- Injectors --------------------
def _find_rank_file_text() -> tuple[str | None, str | None]:
    root = ROOT; data_dir = root / "data"
    patterns = ["nttv rank requirements.txt","rank requirements.txt","*rank*requirements*.txt"]
    search_dirs = [data_dir, root]; seen = set()
    for d in search_dirs:
        if not d.exists(): continue
        for pat in patterns:
            for p in d.glob(pat):
                lp = str(p).lower()
                if lp in seen: continue
                seen.add(lp)
                try:
                    txt = p.read_text(encoding="utf-8", errors="replace")
                    if txt and "kyu" in txt.lower(): return txt, str(p)
                except Exception: pass
    return None, None

def _gather_full_text_for_source(name_contains: str) -> tuple[str | None, str | None]:
    want = name_contains.lower()
    matched = [c for c in CHUNKS if want in os.path.basename((c.get("meta", {}) or {}).get("source", "")).lower()]
    if not matched: return None, None
    parts = [c.get("text") or "" for c in matched]
    full = "\n\n".join(parts).strip() if parts else None
    any_path = matched[0].get("meta", {}).get("source")
    return (full if full else None), any_path

def inject_rank_passage_if_needed(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if "kyu" not in question.lower(): return hits
    txt, path = _gather_full_text_for_source("nttv rank requirements")
    if not txt:
        txt, path = _find_rank_file_text()
    if not txt: return hits
    synth = {
        "text": txt,
        "meta": {"priority": 1, "source": path or "NTTV Rank Requirements (synthetic)"},
        "source": path or "NTTV Rank Requirements (synthetic)",
        "page": None, "score": 1.0, "rerank_score": 999.0,
    }
    return [synth] + hits

def inject_leadership_passage_if_needed(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ql = question.lower()
    if not any(t in ql for t in ["soke","sōke","grandmaster","headmaster","current head","current grandmaster"]):
        return hits
    txt, path = _gather_full_text_for_source("leadership")
    if not txt: return hits
    synth = {
        "text": txt,
        "meta": {"priority": 1, "source": path or "Bujinkan Leadership (synthetic)"},
        "source": path or "Bujinkan Leadership (synthetic)",
        "page": None, "score": 1.0, "rerank_score": 998.0,
    }
    return [synth] + hits

def inject_schools_passage_if_needed(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ql = question.lower()
    triggers = ["ryu","ryū","school","schools","togakure","gyokko","koto",
                "shinden fudo","kukishinden","takagi yoshin","gikan","gyokushin","kumogakure"]
    if not any(t in ql for t in triggers): return hits
    txt, path = _gather_full_text_for_source("schools of the bujinkan summaries")
    if not txt: return hits
    synth = {
        "text": txt,
        "meta": {"priority": 1, "source": path or "Schools of the Bujinkan Summaries (synthetic)"},
        "source": path or "Schools of the Bujinkan Summaries (synthetic)",
        "page": None, "score": 1.0, "rerank_score": 997.0,
    }
    return [synth] + hits

# -------------------- Prompting / LLM --------------------
STRICT_SYSTEM = (
    "You are the NTTV assistant. Answer ONLY from the provided context.\n"
    "Write 2–4 short declarative sentences that quote key phrases from the context when possible.\n"
    "Avoid generic fillers like 'is a style of martial arts', 'is known for unique training', "
    "'originated in Japan', or organizational trivia unless explicitly in context."
)

BANNED_GENERIC = [
    "is a style of martial arts",
    "originated in japan",
    "secluded organization",
    "unique training methods",
    "significant force in the shinobi world",
    "international ninjutsu organization",
]

def build_explanation_prompt(question: str, passages: List[Dict[str, Any]], fact_sentence: str | None) -> str:
    ctx = "\n\n".join(p["text"] for p in passages[:8])
    ban = "; ".join(BANNED_GENERIC)
    if fact_sentence:
        return (
            "Using ONLY the provided context, write 2–4 short declarative sentences.\n"
            "Begin with this exact sentence (do not modify it):\n"
            f"{fact_sentence}\n\n"
            "Then add 1–3 sentences that quote concrete phrases from the context; "
            "avoid generic claims or organizational trivia.\n"
            f"Do NOT use these phrases: {ban}\n\n"
            f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx}\n\nEXPLANATION:\n"
        )
    else:
        return (
            "Using ONLY the provided context, write 2–4 short declarative sentences that answer the question. "
            "Quote concrete phrases from the context; avoid generic claims or organizational trivia.\n"
            f"Do NOT use these phrases: {ban}\n\n"
            f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx}\n\nEXPLANATION:\n"
        )

def clean_answer(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^(here( is|'s) (the )?answer[:,]?\s*)", "", s, flags=re.I)
    s = re.sub(r"^(based on|according to) (the )?(provided )?context[:,]?\s*", "", s, flags=re.I)
    s = re.sub(r"(?m)^\s*(?:[-*]\s+|\[\d+\]\s*)", "", s)
    s = re.sub(r"\[(?:\d+|[A-Za-z])\]", "", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def looks_generic(s: str) -> bool:
    low = (s or "").lower()
    if len(low) < 20: return True
    return any(p in low for p in BANNED_GENERIC)

def call_model_with_fallback(client: OpenAI, model: str, system: str, user: str,
                             max_tokens: int = 512, temperature: float = 0.2) -> tuple[str, str]:
    import json as _json
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

# -------------------- School-section context helper --------------------
def _school_section_only(question: str, hits: list[dict]) -> str | None:
    try:
        from extractors.schools import _target_key_from_question, _best_schools_blob, _slice_single_school
        target = _target_key_from_question(question)
        if not target: return None
        blob = _best_schools_blob(hits) or ""
        sec = _slice_single_school(blob, target)
        return sec
    except Exception:
        return None

def _best_concrete_from_section(section: str) -> str:
    # take the first 3–4 sentences that look concrete (avoid school names list)
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9‘“])", (section or "").strip())
    sents = [s.strip() for s in sents if len(s.strip()) > 5]
    out = []
    for s in sents:
        low = s.lower()
        if any(k in low for k in ["ninpo","ninjutsu","taijutsu","kosshijutsu","koppojutsu",
                                  "dakentaijutsu","jutaijutsu","stealth","infiltration",
                                  "distance","timing","kamae","footwork","weapons","strategy"]):
            out.append(s)
        if len(out) >= 4: break
    return " ".join(out) if out else " ".join(sents[:3])

# -------------------- RAG pipeline --------------------
def answer_with_rag(question: str):
    k = TOP_K
    if "kyu" in question.lower():
        k = max(TOP_K * 4, TOP_K + 10)

    hits = retrieve(question, k=k)
    hits = inject_rank_passage_if_needed(question, hits)
    hits = inject_leadership_passage_if_needed(question, hits)
    hits = inject_schools_passage_if_needed(question, hits)

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    asking_soke = any(t in question.lower() for t in ["soke","sōke","grandmaster","headmaster","current head","current grandmaster"])
    if asking_soke:
        fact = None
        try: fact = try_leadership(question, hits)
        except Exception: fact = None
        if fact:
            return f"🔒 Strict (context-only, explain)\n\n{fact}", hits, '{"det_path":"leadership/soke"}'

    # Deterministic rank/kyusho/etc.
    fact = try_extract_answer(question, hits)

    # NEW: deterministic school profile
    school_fact = try_answer_school_profile(question, hits)
    if school_fact:
        sec = _school_section_only(question, hits)
        if sec:
            explain_hits = [{"text": sec, "meta": {"priority": 1}, "source": "Schools (section)"}]
        else:
            explain_hits = hits

        user = build_explanation_prompt(question, explain_hits, fact_sentence=school_fact)
        content, raw = call_model_with_fallback(
            client=client, model=MODEL, system=STRICT_SYSTEM,
            user=user, max_tokens=min(MAX_TOKENS, 260), temperature=0.0
        )
        content = clean_answer(content) if content else school_fact

        # Anti-bland guard: fall back to deterministic section if the model is generic
        if looks_generic(content) and sec:
            concrete = _best_concrete_from_section(sec)
            content = f"{school_fact}\n{concrete}"

        return f"🔒 Strict (context-only, explain)\n\n{content}", explain_hits, raw

    # If we had a deterministic fact (e.g., kihon, kyusho), explain with broader context
    if fact:
        explain_hits = hits
        user = build_explanation_prompt(question, explain_hits, fact_sentence=fact)
        content, raw = call_model_with_fallback(
            client=client, model=MODEL, system=STRICT_SYSTEM,
            user=user, max_tokens=min(MAX_TOKENS, 260), temperature=0.0
        )
        content = clean_answer(content) if content else fact
        return f"🔒 Strict (context-only, explain)\n\n{content}", explain_hits, raw

    # Last resort: general explanation from context
    explain_hits = hits
    user = build_explanation_prompt(question, explain_hits, fact_sentence=None)
    content, raw = call_model_with_fallback(
        client=client, model=MODEL, system=STRICT_SYSTEM,
        user=user, max_tokens=min(MAX_TOKENS, 260), temperature=0.0
    )
    content = clean_answer(content) if content else content
    # Anti-bland guard (fallback to top passage if needed)
    if looks_generic(content):
        top_text = hits[0]["text"] if hits else ""
        concrete = _best_concrete_from_section(top_text) if top_text else ""
        if concrete:
            content = concrete
    return f"🔒 Strict (context-only, explain)\n\n{content if content else '❌ Model returned no text.'}", explain_hits, raw

# -------------------- UI --------------------
st.set_page_config(page_title="NTTV Chat", page_icon="💬")
st.title("💬 NTTV Chat (Local RAG)")
st.caption("Always explain: strict, context-only answers with deterministic fallbacks.")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("Status")
    st.write(f"Model: `{MODEL}`")
    st.write(f"Server: `{BASE_URL}`")
    st.write(f"Top K: {TOP_K}  |  Max tokens: {MAX_TOKENS}")
    st.markdown("---")
    show_debug = st.checkbox("Show debugging info (sources & raw model)", value=False)
    st.markdown("---")
    st.write("Tip: update your data in `/data`, run `python ingest.py`, then refresh.")

q = st.text_input("Ask a question:", placeholder="e.g., Who is the sōke of Koto-ryū? or Tell me about Gyokko-ryū")
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
                if prio: suffix += f" — priority {prio}"
                if p.get("page"): suffix += f" — p.{p['page']}"
                st.write(base + suffix)
            with st.expander("Show raw model response"):
                st.code(raw_json, language="json")
