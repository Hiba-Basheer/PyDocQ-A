"""
Build and persist a FAISS index from text documents.

This script:
1. Loads text documents from disk
2. Chunks the documents into plain strings
3. Creates embeddings
4. Builds a FAISS index
5. Saves the index and chunks for later retrieval
"""

import logging
import os
import pickle
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
# Configuration
ARTIFACT_DIR = "artifacts"
INDEX_PATH = os.path.join(ARTIFACT_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(ARTIFACT_DIR, "chunks.pkl")
DATA_DIR = "data/python_docs"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)
# Utilities
def load_txt_documents(data_dir: str) -> List[str]:
    """Load all .txt files as raw strings."""
    texts = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            path = os.path.join(data_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())

    return texts


def chunk_text(text: str) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def chunk_documents(documents: List[str]) -> List[str]:
    """Chunk all documents into a flat list of strings."""
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc))
    return all_chunks

# Main
def main() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    LOGGER.info("Loading documents from %s", DATA_DIR)
    documents = load_txt_documents(DATA_DIR)

    LOGGER.info("Chunking documents")
    chunks = chunk_documents(documents)

    LOGGER.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    LOGGER.info("Creating embeddings")
    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    LOGGER.info("Building FAISS index")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    LOGGER.info("Saving FAISS index")
    faiss.write_index(index, INDEX_PATH)

    LOGGER.info("Saving chunks (plain text)")
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    LOGGER.info("Index build complete")


if __name__ == "__main__":
    main()
