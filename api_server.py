# api_server.py
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Single FastAPI app (do NOT re-create later)
app = FastAPI(title="NTTV Chatbot API", version="1.0.1")

# --- CORS for local dev ---
# Wide-open for local testing. In production, replace with a strict allowlist.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # or set explicit origins in prod
    allow_credentials=False,      # must be False when allow_origins=["*"]
    allow_methods=["*"],          # includes OPTIONS (preflight)
    allow_headers=["*"],          # Content-Type, X-API-Key, etc.
)

# Re-use your existing logic (no Streamlit UI runs here)
from app import _load_index_and_meta, answer_with_rag  # noqa: E402

API_KEY = os.getenv("NTTV_API_KEY")  # optional; set in .env or Render
MODEL = os.getenv("MODEL", "google/gemma-3n-e4b-it")


class QueryReq(BaseModel):
    query: str
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class SourceItem(BaseModel):
    source: str
    page: Optional[int] = None
    snippet: str
    score: Optional[float] = None


class QueryResp(BaseModel):
    answer: str
    sources: List[SourceItem]
    det_path: Optional[str] = None
    meta: Dict[str, Any]


@app.get("/")
def root():
    return {"status": "ok", "service": "nttv-chatbot-api", "version": "1.0.1"}


@app.get("/healthz")
def healthz():
    idx, chunks = _load_index_and_meta()
    return {
        "status": "ok",
        "faiss_ntotal": int(getattr(idx, "ntotal", 0) or 0),
        "chunks": len(chunks),
        "model": MODEL,
    }


@app.post("/query", response_model=QueryResp)
def query(req: QueryReq, x_api_key: Optional[str] = Header(default=None)):
    # Optional API key check
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    t0 = time.perf_counter()

    # Full RAG pipeline (includes deterministic extractors)
    answer, hits, det_json = answer_with_rag(req.query, k=req.top_k)

    # Extract det_path from the JSON-ish payload
    det_path = None
    if det_json:
        try:
            if isinstance(det_json, dict):
                det_path = det_json.get("det_path")
            else:
                import json as _json
                det_path = _json.loads(det_json).get("det_path")
        except Exception:
            det_path = None

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    sources: List[SourceItem] = []
    for h in hits or []:
        sources.append(
            SourceItem(
                source=h.get("source") or (h.get("meta") or {}).get("source") or "",
                page=h.get("page"),
                snippet=h.get("text", "") or "",
                score=h.get("rerank_score") or h.get("score"),
            )
        )

    return {
        "answer": answer or "",
        "sources": sources,
        "det_path": det_path,
        "meta": {
            "model": MODEL,
            "retrieval_count": len(hits or []),
            "elapsed_ms": elapsed_ms,
        },
    }
