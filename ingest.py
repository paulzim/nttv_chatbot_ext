import os, json, pickle, sys
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from pypdf import PdfReader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss, re, numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------- Paths (RELATIVE to this repo) ----------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"          # <--- ensure your files are here
INDEX_DIR = ROOT / "index"
INDEX_DIR.mkdir(exist_ok=True, parents=True)

# Required canonical file (so we fail fast if missing)
REQUIRED_FILES = {
    "nttv rank requirements.txt",
}

# ---------- Chunking / Embeddings ----------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# ---------- Filename → priority ----------
PRIORITY_RULES = {
    "nttv rank requirements.txt": 3,       # top
    "nttv training reference.txt": 2,      # next
    "technique descriptions.txt": 2,       # next
}
DEFAULT_PRIORITY = 1

KYU_RE = re.compile(r"\b(10th|9th|8th|7th|6th|5th|4th|3rd|2nd|1st)\s+kyu\b", re.I)

def file_priority(p: Path) -> int:
    return PRIORITY_RULES.get(p.name.lower(), DEFAULT_PRIORITY)

def read_file(path: Path):
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        out = []
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            out.append((txt, i+1))
        return out
    else:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        return [(txt, None)]

def extract_kyu_label(text: str):
    m = KYU_RE.search(text)
    return m.group(0).lower() if m else None

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    out = []
    for doc in docs:
        text, meta = doc["text"], doc["meta"]
        chunks = splitter.split_text(text)
        for i, ch in enumerate(chunks):
            out.append({
                "text": ch,
                "meta": {**meta, "chunk": i, "kyu": extract_kyu_label(ch)},
            })
    return out

def main():
    if not DATA_DIR.exists():
        print(f"ERROR: data folder not found at {DATA_DIR}")
        sys.exit(1)

    # Gather files
    files = [f for f in DATA_DIR.glob("**/*")
             if f.is_file() and f.suffix.lower() in {".txt", ".md", ".markdown", ".pdf"}]

    if not files:
        print(f"ERROR: No documents found in {DATA_DIR}. Add TXT/MD/PDF files and run again.")
        sys.exit(1)

    # Show what we found (helps catch path mistakes)
    print("Found files in data/:")
    for f in sorted(files):
        print(" -", f.name)

    # Enforce presence of required canonical file(s)
    found_lower = {f.name.lower() for f in files}
    missing = [req for req in REQUIRED_FILES if req not in found_lower]
    if missing:
        print("\nERROR: Missing required file(s) in data/:")
        for m in missing: print(" -", m)
        print("Please add them to:", DATA_DIR)
        sys.exit(1)

    # Read and prepare docs
    raw_docs = []
    for f in files:
        prio = file_priority(f)
        base = f.name
        for text, page in read_file(f):
            if text.strip():
                raw_docs.append({
                    "text": text,
                    "meta": {
                        "source": str(f),
                        "source_basename": base,
                        "page": page,
                        "priority": prio
                    }
                })

    # Chunk
    chunks = chunk_docs(raw_docs)
    print(f"\nTotal chunks: {len(chunks)}")

    # Embed
    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c["text"] for c in chunks]

    embs = []
    for i in tqdm(range(0, len(texts), 64), desc="Embedding"):
        batch = texts[i:i+64]
        e = model.encode(batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        embs.append(e)
    X = np.vstack(embs).astype("float32")

    # FAISS index (cosine via IP on normalized vectors)
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    with open(INDEX_DIR / "meta.pkl", "wb") as f:
        pickle.dump(chunks, f)

    with open(INDEX_DIR / "config.json", "w") as f:
        json.dump({"embed_model": EMBED_MODEL_NAME, "dim": dim}, f)

    # Helpful summary
    names = [c["meta"].get("source_basename","").lower() for c in chunks]
    print("\nTop files in index:", Counter(names).most_common(10))
    print("\nIndex built. Files saved in /index.")

if __name__ == "__main__":
    main()
