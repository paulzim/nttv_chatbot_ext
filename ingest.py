"""
Ingest content files, chunk them, embed them, and build a FAISS index.

Usage (locally):
    python ingest.py

On Render:
    Make sure this runs in the build command so the index exists
    before the app starts.
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------
# Paths & constants
# ---------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"          # <--- ensure your files are here

# Index directory:
# - Locally: defaults to <repo>/index
# - On Render (or other hosts): set INDEX_DIR env var, e.g. /var/data/index
DEFAULT_INDEX_DIR = ROOT / "index"
INDEX_DIR = Path(os.getenv("INDEX_DIR") or DEFAULT_INDEX_DIR)

INDEX_DIR.mkdir(exist_ok=True, parents=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CONFIG_PATH = INDEX_DIR / "config.json"
META_PATH = INDEX_DIR / "meta.pkl"
FAISS_PATH = INDEX_DIR / "index.faiss"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120


# ---------------------------
# Utilities
# ---------------------------

def read_text_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        from pypdf import PdfReader  # lazy import
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix in [".docx"]:
        from docx import Document  # type: ignore
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def iter_source_files() -> List[Path]:
    exts = {".txt", ".md", ".pdf", ".docx"}
    files: List[Path] = []
    for p in DATA_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files


def simple_chunk_text(text: str, source: str) -> List[Dict[str, Any]]:
    """Naive character-based chunking with overlap."""

    chunks: List[Dict[str, Any]] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + CHUNK_SIZE, n)
        chunk_text = text[start:end].strip()
        if chunk_text:
            # Heuristic priority: glossaries, rank, technique descriptions get higher
            lower_source = source.lower()
            if "glossary" in lower_source:
                priority = 3
            elif "rank" in lower_source:
                priority = 3
            elif "technique description" in lower_source or "technique_descriptions" in lower_source:
                priority = 3
            elif "kihon" in lower_source or "sanshin" in lower_source:
                priority = 2
            else:
                priority = 1

            chunks.append(
                {
                    "text": chunk_text,
                    "source": source,
                    "meta": {"priority": priority},
                }
            )

        if end == n:
            break
        start = end - CHUNK_OVERLAP

    return chunks


# ---------------------------
# Embeddings & index build
# ---------------------------

def embed_chunks(
    model: SentenceTransformer, chunks: List[Dict[str, Any]]
) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    emb = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    return emb.astype("float32")


def build_faiss_index(
    embeddings: np.ndarray,
) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def main() -> None:
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"INDEX_DIR: {INDEX_DIR}")

    files = iter_source_files()
    if not files:
        raise RuntimeError(f"No source files found in {DATA_DIR}")

    print("Found source files:")
    for f in files:
        print(" -", f.relative_to(ROOT))

    all_chunks: List[Dict[str, Any]] = []
    for f in files:
        print(f"\nReading {f} ...")
        text = read_text_file(f)
        print(f"  Length: {len(text)} characters")
        cks = simple_chunk_text(text, source=str(f.relative_to(ROOT)))
        print(f"  -> {len(cks)} chunks")
        all_chunks.extend(cks)

    print(f"\nTotal chunks: {len(all_chunks)}")

    print("\nLoading embedding model:", EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Embedding chunks...")
    emb = embed_chunks(model, all_chunks)
    print("Embeddings shape:", emb.shape)

    print("Building FAISS index...")
    index = build_faiss_index(emb)

    print(f"Saving FAISS index to {FAISS_PATH}")
    faiss.write_index(index, str(FAISS_PATH))

    print(f"Saving metadata to {META_PATH}")
    with META_PATH.open("wb") as f:
        pickle.dump(all_chunks, f)

    config = {
        "embed_model": EMBED_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "files": [str(f.relative_to(ROOT)) for f in files],
    }
    print(f"Saving config to {CONFIG_PATH}")
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"\nIndex built. Files saved in {INDEX_DIR}")


if __name__ == "__main__":
    main()
