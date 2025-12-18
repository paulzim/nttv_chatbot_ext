# NTTV Chatbot — Deterministic RAG Assistant for Ninja Training TV

A **RAG-based**, **extractor-driven**, and **deterministic** chatbot for Ninja Training TV (NTTV).  
Built with **FAISS**, **sentence-transformers**, **Streamlit**, and a suite of custom **extractors** (rank, kihon, sanshin, schools, weapons, kyusho, etc.).  
Runs **locally** and on **Render** with the same ingestion + retrieval pipeline.

---

## Table of Contents
- [Key Features](#-key-features)
- [Architecture (Brief)](#-architecture-brief)
- [Repository Structure](#-repository-structure)
- [Installation (Local)](#-installation-local)
- [Environment Variables](#-environment-variables)
- [Build the Index & Run Streamlit](#-build-the-index--run-streamlit)
- [Accessing the NTTV Chatbot API (Local)](#accessing-the-nttv-chatbot-api-local)
- [Deploying to Render](#-deploying-to-render)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [License & Credits](#-license--credits)

---

## 🚀 Key Features

### 🧠 Deterministic Knowledge Layer (Extractors)
- Extractors for:
  - **Rank requirements**
  - **Kihon Happō**
  - **Sanshin no Kata**
  - **Schools (Ryūha)**
  - **Weapons**
  - **Kyusho**
- Hard-coded, rank-aware answers where appropriate.
- **Zero hallucinations** for strict/deterministic queries when extractors fire.
- UI badge via `{"det_path":"deterministic/..."}` when a deterministic path answered.

### 🔍 RAG Retrieval Engine
- **FAISS** vector index, **IndexFlatIP** (exact) on **normalized** 384-dim vectors.
- **Sentence-Transformers** embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
- Priority-aware reranking:
  - **P1**: Rank files
  - **P2**: Techniques / schools / kihon / weapons
  - **P3**: Other passages
- Adjustable **TOP_K** and robust fallback heuristics.

### 💬 Streamlit App UI
- Question input + answer display.
- **Debug mode** (top passages, raw model response).
- **Explanation mode** (short fact → brief rationale).
- Technique detail level (Brief / Standard / Full).
- Source citations and passage inspection.

---

## 🧱 Architecture (Brief)

**Ingestion (`ingest.py`)**
- Reads `/data`, chunks, embeds, and writes artifacts to `/index`.
- Artifacts:
  - `index/faiss.index` **and** `index/index.faiss` (dual-write to avoid env drift)
  - `index/meta.pkl` (list of chunk dicts)
  - `index/config.json` (paths, counts, model, chunking params)
- Strict pre/post checks ensure **1:1 alignment** between FAISS vectors and `meta.pkl`.

**App / Retrieval (`app.py`)**
- Lazy cached loader `_load_index_and_meta()` with sanity checks:
  - Rejects mismatched FAISS/meta pairs and falls back appropriately.
- Retrieval → overfetch (capped by `ntotal`) → rerank → build context → LLM.
- Deterministic extractors run first for “known-knowns”; fallback to RAG+LLM.

**LLM**
- OpenRouter-compatible endpoint via `OPENAI_BASE_URL` + `OPENAI_API_KEY`.
- Switch model with env `MODEL` (e.g., `google/gemma-3n-e4b-it`).

---

## 📦 Repository Structure

```
nttv_chatbot_ext/
├── app.py                 # Streamlit UI + retrieval + extractors routing
├── ingest.py              # Build FAISS + meta from /data
├── api_server.py          # (Optional) FastAPI server for /query
├── extractors/            # Deterministic extractors (rank, kihon, weapons, etc.)
├── data/                  # Authoritative text sources
├── index/                 # FAISS artifacts (created by ingest.py)
├── requirements.txt       # Python dependencies
├── render.yaml            # Render Blueprint for cloud deployment
├── nttv_test.html         # Minimal chat test page (vanilla HTML/JS)
└── README.md              # You are here
```

---

## 🛠 Installation (Local)

### 1) Clone & create venv
```bash
git clone https://github.com/paulzim/nttv_chatbot_ext
cd nttv_chatbot_ext
python -m venv .venv
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```bash
python -m pip install -U pip
pip install -r requirements.txt
```

---

## ⚙️ Environment Variables

Used locally (via `.env`) and on Render.

| Variable                               | Example                                   | Purpose                                  |
|----------------------------------------|-------------------------------------------|------------------------------------------|
| `OPENAI_BASE_URL`                      | `https://openrouter.ai/api/v1`            | OpenRouter/OpenAI-compatible endpoint    |
| `OPENAI_API_KEY`                       | `sk-or-...`                               | API key (keep secret)                    |
| `MODEL`                                | `google/gemma-3n-e4b-it`                  | LLM model ID                             |
| `EMBED_MODEL_NAME`                     | `sentence-transformers/all-MiniLM-L6-v2`  | Embedding model                          |
| `INDEX_DIR`                            | `index`                                   | Index directory root                     |
| `INDEX_PATH`                           | `index/faiss.index`                       | FAISS index file path (dual-written)     |
| `META_PATH`                            | `index/meta.pkl`                          | Pickled metadata (chunks)                |
| `RANK_FILE`                            | `data/nttv rank requirements.txt`         | Rank source of truth                     |
| `TOP_K`                                | `6`                                       | Retrieval depth                          |
| `TEMPERATURE`                          | `0.0`                                     | Deterministic output                     |
| `MAX_TOKENS`                           | `512`                                     | Generation token cap                     |
| `STREAMLIT_BROWSER_GATHERUSAGESTATS`   | `false`                                   | Disable Streamlit telemetry              |

> **Note:** Historical configs may use `MODEL_NAME`; this project expects `MODEL`.

**Example `.env` (local)**
```env
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-xxxx
MODEL=google/gemma-3n-e4b-it

EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
INDEX_DIR=index
INDEX_PATH=index/faiss.index
META_PATH=index/meta.pkl
RANK_FILE=data/nttv rank requirements.txt

TOP_K=6
TEMPERATURE=0.0
MAX_TOKENS=512
STREAMLIT_BROWSER_GATHERUSAGESTATS=false
```

> Do **not** commit `.env` or real secrets to git.

---

## 🧪 Build the Index & Run Streamlit

### Build / Rebuild FAISS
```bash
python ingest.py
```
Writes:
- `index/faiss.index` **and** `index/index.faiss`
- `index/meta.pkl`
- `index/config.json`

### Run the Streamlit UI
```bash
streamlit run app.py
```
Open `http://localhost:8501`.

---

## Accessing the NTTV Chatbot API (Local)

This repo includes a small **FastAPI** server (`api_server.py`) so you can hit the RAG pipeline over HTTP, plus a minimal **vanilla HTML** test page (`nttv_test.html`) for quick UI testing.

### Prereqs
- Python 3.11.x
- A virtualenv with project deps installed
- Index artifacts built (`python ingest.py`)

> Tip: Always run Python commands **inside your venv**.

```powershell
# Windows / PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install fastapi uvicorn[standard] pydantic
```

---

### 1) Launch the API server (FastAPI)

```powershell
# From repo root, with venv active
# Optional: set an API key for local auth (omit if you don't want auth)
$env:NTTV_API_KEY = "dev-local-key"

# Start the API on 127.0.0.1:8000 (reload on file changes)
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
```

**Health check:**
```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz
```

**Query example (PowerShell):**
```powershell
# Without API key
Invoke-RestMethod -Uri http://127.0.0.1:8000/query -Method Post -ContentType 'application/json' -Body '{"query":"When do I learn kusari-fundo?"}'

# With API key
Invoke-RestMethod -Uri http://127.0.0.1:8000/query -Method Post -ContentType 'application/json' -Headers @{ 'X-API-Key' = $env:NTTV_API_KEY } -Body '{"query":"When do I learn kusari-fundo?"}'
```

**Query example (curl.exe):**
```bat
curl.exe -s -H "Content-Type: application/json" ^
  -H "X-API-Key: dev-local-key" ^
  -d "{\"query\":\"When do I learn kusari-fundo?\"}" ^
  http://127.0.0.1:8000/query
```

**Response shape:**
```json
{
  "answer": "string",
  "sources": [
    { "source": "file.md", "page": null, "snippet": "…", "score": 0.43 }
  ],
  "det_path": "deterministic/kihon",
  "meta": {
    "model": "google/gemma-3n-e4b-it",
    "retrieval_count": 6,
    "elapsed_ms": 812
  }
}
```

> **CORS:** `api_server.py` ships with dev-friendly CORS (wide-open). For production, restrict `allow_origins` to your site domain.

---

### 2) Launch the minimal test webpage (no framework)

We ship a tiny, framework-free test page: `nttv_test.html`. Serve it locally to avoid `file://` CORS issues.

```powershell
# In the folder containing nttv_test.html
python -m http.server 5500
# Open in your browser:
# http://127.0.0.1:5500/nttv_test.html
```

**Point the page at your API (one-time):**  
Open the browser DevTools **Console** on `nttv_test.html` and run:
```js
localStorage.setItem("NTTV_API_BASE","http://127.0.0.1:8000");
// If you set an API key when launching uvicorn, also set:
localStorage.setItem("NTTV_API_KEY","dev-local-key");
location.reload();
```

Now type a question in the page and you’ll see:
- The assistant’s answer as a chat bubble
- A **Deterministic** badge when a rule-based extractor answered
- Numbered citations (sources + snippets)
- Basic timing in the metadata line

---

## ☁️ Deploying to Render

1) Ensure `render.yaml` is at the repo root (Blueprint).  
2) In Render:
   - New → **Blueprint** → connect repo.
   - `buildCommand`:
     ```bash
     python -m pip install -U pip && pip install -r requirements.txt && python ingest.py
     ```
   - `startCommand` (Streamlit):
     ```bash
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0
     ```
   - Set env vars in the dashboard:
     - `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `MODEL`
     - `INDEX_DIR`, `INDEX_PATH`, `META_PATH` (if you override)
     - Optional: `TRANSFORMERS_CACHE`, `SENTENCE_TRANSFORMERS_HOME` on persistent disk
   - Mount a persistent disk for `/var/data/index` if desired; point `INDEX_DIR` there.

> Free tiers are RAM-limited; for larger corpora, pick a plan with ≥2 GB RAM.

---

## 🔧 Troubleshooting

**“Index/meta mismatch” or retrieval errors**
- Rebuild: delete `index/` and run `python ingest.py`  
- Ensure **counts match** in `config.json` and sidebar diagnostics.

**CORS blocked in browser**
- Use the served page (`http://127.0.0.1:5500/...`).
- Keep dev CORS in `api_server.py`; restrict origins for production.

**401 Unauthorized**
- You started uvicorn with `NTTV_API_KEY`. Send `X-API-Key` from the client or unset the env var and restart.

**LLM errors (401/403/429)**
- Check `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `MODEL`.

**Slow or 5xx on Render**
- Free tier may sleep; cold starts add latency.
- Upgrade to more RAM/CPU if you see restarts or OOM.

---

## 🧭 Roadmap
- Add deterministic extractor for **Kyusho** (expanded).
- Optional HNSW index (env-gated) for faster retrieval at scale.
- SSE/WebSocket streaming endpoint for token-by-token UX.
- Simple `deploy.sh` for VPS targets.
- Expand prompt harness with rank/weapons technique cases.

---

## 📜 License & Credits

**License:** MIT

**Built with:**
- Streamlit  
- FAISS  
- Sentence-Transformers  
- OpenRouter-compatible APIs  
- NTTV curriculum (Bujinkan-focused content)

