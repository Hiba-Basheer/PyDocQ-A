"""
Build and persist a FAISS index from text documents.

Pipeline:
1. Load .txt documents
2. Chunk documents into overlapping strings
3. Generate sentence embeddings
4. Build FAISS index
5. Persist index and chunks for retrieval
"""

import logging
import os
import pickle
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Configuration

DATA_DIR = "data/python_docs"
ARTIFACT_DIR = "artifacts"

INDEX_PATH = os.path.join(ARTIFACT_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(ARTIFACT_DIR, "chunks.pkl")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Document loading & chunking

def load_txt_documents(data_dir: str) -> List[str]:
    """
    Load all .txt files from a directory as raw strings.
    """
    LOGGER.info("Loading .txt documents from %s", data_dir)
    documents: List[str] = []

    for filename in os.listdir(data_dir):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(data_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            documents.append(f.read())

    LOGGER.info("Loaded %d documents", len(documents))
    return documents


def chunk_text(text: str) -> List[str]:
    """
    Split a single document into overlapping chunks.
    """
    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def chunk_documents(documents: List[str]) -> List[str]:
    """
    Chunk all documents into a flat list of text chunks.
    """
    LOGGER.info("Chunking documents")
    all_chunks: List[str] = []

    for doc in documents:
        all_chunks.extend(chunk_text(doc))

    LOGGER.info("Created %d chunks", len(all_chunks))
    return all_chunks

# Indexing

def build_faiss_index(chunks: List[str]) -> faiss.Index:
    """
    Create embeddings and build a FAISS index.
    """
    LOGGER.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    LOGGER.info("Encoding %d chunks", len(chunks))
    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    LOGGER.info("Building FAISS index")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    LOGGER.info("FAISS index contains %d vectors", index.ntotal)
    return index

# Persistence

def save_artifacts(index: faiss.Index, chunks: List[str]) -> None:
    """
    Save FAISS index and chunks to disk.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    LOGGER.info("Saving FAISS index to %s", INDEX_PATH)
    faiss.write_index(index, INDEX_PATH)

    LOGGER.info("Saving chunks to %s", CHUNKS_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

# Entry point

def main() -> None:
    documents = load_txt_documents(DATA_DIR)
    chunks = chunk_documents(documents)
    index = build_faiss_index(chunks)
    save_artifacts(index, chunks)

    LOGGER.info("Index build complete")

if __name__ == "__main__":
    main()
