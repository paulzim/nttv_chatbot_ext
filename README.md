# NTTV Chatbot — Deterministic RAG Assistant for Ninja Training TV

A **RAG-based**, **extractor-driven**, and **deterministic** chatbot for Ninja Training TV (NTTV).  
Built in Python using **FAISS**, **sentence-transformers**, **Streamlit**, and a suite of custom **extractors** for rank, kihon, sanshin, schools, weapons, kyusho, and more.

Runs **locally** or in the **cloud (Render)** with the same index + retrieval pipeline.

---

## 🚀 Key Features

### 🧠 Deterministic Knowledge Layer

- Extractors for:
  - **Rank requirements**
  - **Kihon Happō**
  - **Sanshin no Kata**
  - **Schools (Ryūha)**
  - **Weapons**
  - **Kyusho**
- Hard-coded, rank-aware responses where appropriate.
- Zero hallucinations for strict/deterministic queries when extractors fire.

### 🔍 RAG Retrieval Engine

- **FAISS** vector index
- **Sentence-Transformers** embeddings (`all-MiniLM-L6-v2`)
- Priority-aware reranking:
  - **P1**: Rank files
  - **P2**: Techniques / schools / kihon / weapons
  - **P3**: Other passages
- Adjustable **TOP_K** and fallback heuristics.

### 💬 Streamlit App UI

- Question input + answer display
- **Debug mode** (shows top passages, raw model response)
- **Explanation mode** (short fact → brief rationale)
- **Technique detail level** (Brief / Standard / Full)
- Source citations and passage inspection

---

## 📦 Repository Structure

    nttv_chatbot_ext/
    │
    ├── app.py                 # Streamlit UI + RAG pipeline
    ├── ingest.py              # Builds FAISS index + meta.pkl from /data
    ├── extractors/            # Deterministic extractors (rank, kihon, weapons, etc.)
    ├── data/                  # Authoritative text sources
    ├── index/                 # Local FAISS index artifacts (created by ingest.py)
    ├── tests/                 # Pytest suite + prompt harness
    ├── requirements.txt       # Python dependencies
    ├── render.yaml            # Render Blueprint for cloud deployment
    └── README.md              # You are here

---

## 🛠 Installation (Local)

### 1. Clone the repo

    git clone https://github.com/paulzim/nttv_chatbot_ext
    cd nttv_chatbot_ext

### 2. Create a virtual environment

macOS / Linux:

    python -m venv .venv
    source .venv/bin/activate

Windows (PowerShell):

    python -m venv .venv
    .\.venv\Scripts\activate

### 3. Install dependencies

    pip install -U pip
    pip install -r requirements.txt

### 4. Build the FAISS index

    python ingest.py

This reads all files in `data/`, chunks them, embeds them, and writes:

- `index/faiss.index`
- `index/meta.pkl`
- `index/config.json`

### 5. Run the chatbot

    streamlit run app.py

Then open the provided URL (typically `http://localhost:8501`).

---

## ⚙️ Environment Variables

Used locally (via `.env`) and in the cloud (via Render).

| Variable              | Example                                         | Purpose                               |
|-----------------------|-------------------------------------------------|---------------------------------------|
| `OPENAI_BASE_URL`     | `https://openrouter.ai/api/v1`                 | Endpoint for model inference          |
| `OPENAI_API_KEY`      | `sk-or-...`                                    | API key (keep secret)                 |
| `MODEL_NAME`          | `openrouter/anthropic/claude-3.5-sonnet`       | LLM identifier                        |
| `EMBED_MODEL_NAME`    | `sentence-transformers/all-MiniLM-L6-v2`       | Embedding model for FAISS             |
| `INDEX_DIR`           | `index/`                                       | Index directory root                  |
| `INDEX_PATH`          | `index/faiss.index`                            | FAISS index file path                 |
| `META_PATH`           | `index/meta.pkl`                               | Metadata (retrieval chunks)           |
| `RANK_FILE`           | `data/nttv rank requirements.txt`              | Rank source of truth                  |
| `TOP_K`               | `6`                                            | Retrieval depth                       |
| `TEMPERATURE`         | `0.0`                                          | Deterministic output                  |
| `MAX_TOKENS`          | `512`                                          | Generation token cap                  |
| `STREAMLIT_BROWSER_GATHER_USAGE_STATS` | `false`                      | Disable Streamlit telemetry           |

### Example `.env` (local)

    OPENAI_BASE_URL=https://openrouter.ai/api/v1
    OPENAI_API_KEY=sk-or-xxxx
    MODEL_NAME=openrouter/anthropic/claude-3.5-sonnet

    EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
    INDEX_DIR=index
    INDEX_PATH=index/faiss.index
    META_PATH=index/meta.pkl
    RANK_FILE=data/nttv rank requirements.txt

    TOP_K=6
    TEMPERATURE=0.0
    MAX_TOKENS=512
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

> ⚠️ Do **not** commit `.env` or real secrets to git.

---

## 🧪 Testing

Run the full test suite:

    pytest -q

Includes:

- Extractor tests (rank, kihon, sanshin, weapons, schools, etc.)
- Retrieval overlap / consistency checks
- Prompt harness using real rank file as first passage
- Technique normalization validations

You can add new prompt tests under:

    tests/prompts/

---

## ☁️ Deploying to Render

### 1. Ensure `render.yaml` is at the repo root

Render uses this as a Blueprint for the service.

### 2. In Render

- New → “Blueprint” → connect this repo.
- Confirm `buildCommand` runs:

    pip install -U pip && pip install -r requirements.txt && python ingest.py

- Confirm `startCommand` is:

    streamlit run app.py --server.port $PORT --server.address 0.0.0.0

- Add environment variables in the Render dashboard:
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL`
  - `MODEL_NAME`
  - Any overrides for `INDEX_DIR`, `TOP_K`, etc.

### 3. Index rebuilds

Whenever files in `data/` change:

- Push a new commit → Render rebuilds and re-runs `python ingest.py`, or
- Use “Manual Deploy → Clear build cache & deploy” to force a fresh ingest.

---

## 🔧 Common Issues

### “Index config / meta not found”

- Make sure `python ingest.py` ran successfully.
- Verify that `index/config.json` and `index/meta.pkl` exist.
- Check `INDEX_DIR`, `INDEX_PATH`, and `META_PATH` in:
  - Local `.env`, and/or
  - Render’s environment settings.

### FAISS index issues

- Ensure `faiss-cpu` is installed (see `requirements.txt`).
- Confirm `faiss.index` path matches `INDEX_PATH` (or config.json’s `faiss_path`).

### LLM errors (401/403/429)

- Check `OPENAI_API_KEY` validity and scope.
- Verify `OPENAI_BASE_URL` is correct (including `/v1` suffix).
- Make sure `MODEL_NAME` is available to your key.

### Slow or 5xx responses on Render

- Free tier may sleep and add startup latency.
- Upgrade to a plan with more RAM/CPU for smoother RAG + Streamlit.
- Use Streamlit’s debug mode to inspect retrieval / model timings.

---

## 🧭 Roadmap

- Add deterministic extractor for **Kyusho**.
- Expand schools and weapons metadata.
- Add `/healthz` or similar health check endpoint.
- Provide a simple `deploy.sh` for DigitalOcean / VPS targets.
- Grow the prompt harness with rank- and weapon-specific test cases.

---

## 📜 License

MIT License — free for personal or commercial use.

---

## 🙏 Credits

Built with:

- Streamlit  
- FAISS  
- Sentence-Transformers  
- OpenRouter / OpenAI-compatible APIs  
- And a lot of Bujinkan / NTTV curriculum work
