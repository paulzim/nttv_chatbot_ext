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

## 📜 **License**
MIT License — free to use, modify, and share.

## 🙋 **Credits**
Developed with:
- LM Studio
- Streamlit
- FAISS
- Sentence Transformers
