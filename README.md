# Local RAG Chatbot with LM Studio + Gemma-3-12B + Streamlit

This project is a **Retrieval-Augmented Generation (RAG) chatbot** that runs **completely locally**.  
It uses:

- **LM Studio** to serve the `gemma-3-12b` model via an OpenAI-compatible API.
- **FAISS** for fast local vector search over your documents.
- **Streamlit** for an interactive web-based chat interface.
- **Sentence Transformers** to embed document chunks and user queries.
- **Reranking & Hybrid Logic** to improve retrieval accuracy.

---

## 🚀 Features

- Runs entirely on your machine — **no external API calls**.
- Loads and indexes local `.txt` documents into FAISS.
- Answers questions using **retrieved document chunks** as context.
- **Hybrid fallback** mode (optional) — can answer from the model’s own knowledge if retrieval fails.
- Shows **source citations** for transparency.
- Simple to extend with PDFs, DOCX, or other formats.

---

## 📂 Project Structure

 ```
 nttv_ai_bot/
 │
 ├── app.py # Streamlit UI + RAG query logic
 ├── ingest.py # Document parsing, chunking, embedding, and FAISS index build
 ├── requirements.txt # Python dependencies
 ├── .env # Environment variables (model name, API URL, etc.)
 ├── index/ # FAISS index files (created by ingest.py)
 ├── docs/ # Source documents (.txt in current setup)
 └── README.md # This file
```

## 🛠 Prerequisites

- **Python 3.10+** (recommend using `venv` or `conda`)
- **LM Studio** installed ([Download here](https://lmstudio.ai/))
- `gemma-3-12b` model downloaded in LM Studio
- **Git** for cloning the repo

---

## 📥 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/nttv_ai_bot.git
   cd nttv_ai_bot
2. Create and activate a virtual environment
python -m venv .venv
 source .venv/bin/activate   # Mac/Linux
 .venv\Scripts\activate      # Windows
3. Install dependencies
 pip install -r requirements.txt
4. Set up environment variables
Create a .env file in the root directory:
env
OPENAI_BASE_URL=http://127.0.0.1:1234   # Local LM Studio API endpoint
OPENAI_API_KEY=lm-studio                # Dummy key (LM Studio ignores it)
MODEL_NAME=google/gemma-3-12b           # Model identifier in LM Studio
## 📄 **Preparing Your Documents**
1. Place your .txt files in the docs/ folder.
2. Run the ingestion script to create the FAISS index:
 python ingest.py
This will:
- Read and chunk your text files.
- Generate embeddings with SentenceTransformers.
- Store them in index/faiss.index and index/meta.pkl.

## 💬 **Running the Chatbot**
1. Start LM Studio:
2. Load gemma-3-12b (or your chosen model).
3. Click Local Server to enable the OpenAI API endpoint.
Note the local address (e.g., http://127.0.0.1:1234).
4. Run Streamlit:
 streamlit run app.py
5. Open the browser window shown in your terminal (usually http://localhost:8501).
6. Ask your question!
- The app retrieves the top-k most relevant document chunks from FAISS.
- Passes them as context to the model.
- Displays the answer + retrieved sources.

## ⚙ **Configuration**
You can tweak these in .env or inside app.py:

Variable	Description	Default
MODEL_NAME	Model identifier in LM Studio	google/gemma-3-12b
OPENAI_BASE_URL	API endpoint for LM Studio	http://127.0.0.1:1234
OPENAI_API_KEY	Dummy key for OpenAI-compatible clients	lm-studio
TOP_K	Number of retrieved chunks	5
MAX_CONTEXT	Max characters from retrieved chunks	2000

## 🔧  **Extending**
Adding PDF Support
- Install PyPDF:
 pip install pypdf
Update ingest.py to check .pdf and parse with PdfReader.
- Adding DOCX Support
Install python-docx:
 pip install python-docx
Update ingest.py with a .docx handler.
- Using a Hosted LLM
Change OPENAI_BASE_URL and MODEL_NAME to your hosted provider’s details.
Keep the RAG logic the same — only the model endpoint changes.

## 🐛 **Troubleshooting**
- FAISS index not found
Run python ingest.py again after adding docs.
- Model returned no text
Ensure LM Studio local server is running and MODEL_NAME matches exactly.
- Slow responses
Reduce MAX_CONTEXT or switch to a smaller model.

# Updates

# 🥋 NTTV Chatbot (Local RAG → Render)

Deterministic, rank-aware RAG chatbot for NTTV. Local-first design with a one-file Render blueprint for hosting.

---

## Features

- **Deterministic extractors** for Rank / Kihon / Sanshin / Schools (+ Kyusho optional)
- **Rank injection**: rank file always visible to the model for any `kyu` query
- **Priority-aware reranker** (P1 rank, P2 training/technique, P3 other)
- **Explanation Mode** toggle (short fact ➜ brief rationale from context)
- **FAISS + sentence-transformers** retrieval
- **Streamlit** UI

---

## Repo Structure (key files)

app.py # Streamlit app (Explanation Mode toggle included)
ingest.py # Builds FAISS index from /data into INDEX_PATH/META_PATH
extractors/ # Deterministic extractors (rank, kihon, sanshin, schools, kyusho)
data/ # Source of truth text files
index/ # (Local) FAISS artifacts — cloud uses a mounted disk
tests/ # Pytest + prompt harness
render.yaml # Render blueprint (this file)

perl
Copy code

---

## Environment Variables

These are used locally (via `.env`) and in the cloud (via Render dashboard or `render.yaml`).

| Key                           | Example / Default                                       | Purpose |
|------------------------------|---------------------------------------------------------|---------|
| `OPENAI_BASE_URL`            | `https://openrouter.ai/api/v1`                          | OpenRouter API root (must end with `/v1`) |
| `OPENAI_API_KEY`             | `sk-or-...`                                             | OpenRouter key (set as secret in Render) |
| `MODEL_NAME`                 | `openrouter/anthropic/claude-3.5-sonnet`               | Model ID via OpenRouter |
| `EMBED_MODEL_NAME`           | `sentence-transformers/all-MiniLM-L6-v2`               | Embedding model for FAISS |
| `INDEX_PATH`                 | `index/faiss.index` (local) or `/var/data/index/faiss.index` (Render) | FAISS index path |
| `META_PATH`                  | `index/meta.pkl` (local) or `/var/data/index/meta.pkl` | Index metadata |
| `RANK_FILE`                  | `data/nttv rank requirements.txt`                      | Primary rank source of truth |
| `TOP_K`                      | `8`                                                     | Retrieval fan-out (auto-boost for kyu queries) |
| `MAX_TOKENS`                 | `512`                                                   | Generation cap |
| `TEMPERATURE`                | `0.0`                                                   | Determinism |
| `WEAK_THRESH`                | `0.35`                                                  | Switch to hybrid when retrieval is weak |
| `STREAMLIT_BROWSER_GATHER_USAGE_STATS` | `false`                                      | Disable Streamlit telemetry |

### Example `.env` for local
```ini
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-...
MODEL_NAME=openrouter/anthropic/claude-3.5-sonnet

EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
INDEX_PATH=index/faiss.index
META_PATH=index/meta.pkl
RANK_FILE=data/nttv rank requirements.txt

TOP_K=8
MAX_TOKENS=512
TEMPERATURE=0.0
WEAK_THRESH=0.35
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
⚠️ Never commit real secrets. Keep .env out of git.

Local Development
bash
Copy code
# Create & activate venv (Windows PowerShell)
py -m venv .venv
.\.venv\Scripts\activate

pip install -U pip
pip install -r requirements.txt

# Build/rebuild FAISS index
python ingest.py

# Run the app
streamlit run app.py
Tests (including prompt harness)
bash
Copy code
pytest -q
Add rank QA prompts under tests/prompts/ (see examples in repo).

The harness injects the real rank file as the first passage so extractors stay deterministic.

Deploy to Render (cloud)
One-time setup
Fork or push this repo to your GitHub.

Ensure render.yaml is at repo root.

In Render: New → Blueprint → connect the repo.

On first deploy, add the secret OPENAI_API_KEY in the service’s Environment tab.

(Optional) Adjust MODEL_NAME, plan, and region.

What the blueprint does
Installs deps.

Runs python ingest.py during build to populate the FAISS index.

Starts Streamlit with --server.port $PORT --server.address 0.0.0.0.

Mounts a persistent disk at /var/data/index so the index survives restarts.

If you change files in /data, Render will rebuild the index on the next deploy.
If you want to rebuild without a code change, hit Manual Deploy → Clear build cache & deploy.

Common Issues & Fixes
502 on first boot
The app may come up before Streamlit binds the port. Render will retry automatically. If it persists, check logs for missing OPENAI_API_KEY.

Model errors (401/403)
Ensure OPENAI_BASE_URL=https://openrouter.ai/api/v1 and the MODEL_NAME is valid for your key.

Index not found
Verify INDEX_PATH and META_PATH point to the mounted disk on Render (/var/data/index/...) and that ingest.py runs in buildCommand.

Slow cold start
First request after deploy may be slower as the platform warms containers and the embeddings model is JIT-loaded.

Ops Tips
Explanation Mode defaults to OFF for determinism. Toggle it in the sidebar when you want brief rationale.

Keep TEMPERATURE=0.0 in production. If you temporarily raise it, do so only with explain=True.

For observability, consider adding structured logging to record: query type (rank/non-rank), extractor hit/miss, retrieval strength, and mode (strict/hybrid/explain).

Roadmap (next)
Add kyusho.py extractor (deterministic, context-only).

Expand rank.py synonyms (Nage, Weapons) and keep them strict.

Add more prompt packs under tests/prompts/ per rank block.

Optional: add a /healthz route or a tiny st.experimental_rerun() heartbeat on Streamlit if you want Render health checks.



## 📜 **License**
MIT License — free to use, modify, and share.

## 🙋 **Credits**
Developed with:
- LM Studio
- Streamlit
- FAISS
- Sentence Transformers
