"""
Create sentence embeddings from text chunks and store them in a FAISS index.

This module loads text documents, chunks them semantically,
creates embeddings, and persists both embeddings and metadata
for downstream retrieval and Q&A.
"""

import logging
import os
import pickle
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from load_and_chunk import load_txt_documents, chunk_documents

# Configuration

DATA_DIR = "data/python_docs"
ARTIFACT_DIR = "artifacts"

INDEX_PATH = os.path.join(ARTIFACT_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(ARTIFACT_DIR, "chunks.pkl")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Logging configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Embedding & indexing

def create_embeddings(
    chunks: List[str],
    model: SentenceTransformer,
) -> np.ndarray:
    """
    Convert text chunks into vector embeddings.

    Args:
        chunks (List[str]): List of text chunks.
        model (SentenceTransformer): Loaded embedding model.

    Returns:
        np.ndarray: Embedding matrix (float32).
    """
    LOGGER.info("Encoding %d chunks into embeddings", len(chunks))

    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    LOGGER.info("Embeddings shape: %s", embeddings.shape)
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index from embeddings.

    Args:
        embeddings (np.ndarray): Embedding matrix.

    Returns:
        faiss.Index: FAISS index.
    """
    LOGGER.info("Building FAISS index")
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    LOGGER.info("FAISS index contains %d vectors", index.ntotal)
    return index


def save_artifacts(index: faiss.Index, chunks: List[str]) -> None:
    """
    Persist FAISS index and chunks to disk.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    LOGGER.info("Saving FAISS index to %s", INDEX_PATH)
    faiss.write_index(index, INDEX_PATH)

    LOGGER.info("Saving chunks to %s", CHUNKS_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

# Script entry point

def main() -> None:
    """
    Full indexing pipeline:
    - load documents
    - chunk semantically
    - embed
    - build FAISS index
    - persist artifacts
    """
    LOGGER.info("Loading documents from %s", DATA_DIR)
    documents = load_txt_documents(DATA_DIR)

    LOGGER.info("Chunking documents")
    chunks = chunk_documents(documents)

    LOGGER.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    embeddings = create_embeddings(chunks, embed_model)
    index = build_faiss_index(embeddings)

    save_artifacts(index, chunks)

    LOGGER.info(
        "Indexing complete | Chunks: %d | Index size: %d",
        len(chunks),
        index.ntotal,
    )


if __name__ == "__main__":
    main()
