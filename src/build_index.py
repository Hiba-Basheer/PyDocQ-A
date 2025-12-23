"""
Build and persist a FAISS index from text documents.

This script:
1. Loads text documents from disk
2. Chunks the documents
3. Creates embeddings
4. Builds a FAISS index
5. Saves the index and chunk metadata for later retrieval
"""

import logging
import os
import pickle

import faiss

from embeddings import create_embeddings
from load_and_chunk import chunk_documents, load_txt_documents

# Configuration

ARTIFACT_DIR = "artifacts"
INDEX_PATH = os.path.join(ARTIFACT_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(ARTIFACT_DIR, "chunks.pkl")
DATA_DIR = "data/python_docs"

# Logging configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


# Main logic

def main() -> None:
    """
    Execute the end-to-end pipeline for building a FAISS index.

    This function orchestrates document loading, chunking, embedding creation,
    FAISS index construction, and persistence to disk.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    LOGGER.info("Loading documents from %s", DATA_DIR)
    documents = load_txt_documents(DATA_DIR)

    LOGGER.info("Chunking documents")
    chunks = chunk_documents(documents)

    LOGGER.info("Creating embeddings (one-time operation)")
    embeddings = create_embeddings(chunks)

    LOGGER.info("Building FAISS index")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    LOGGER.info("Saving FAISS index to %s", INDEX_PATH)
    faiss.write_index(index, INDEX_PATH)

    LOGGER.info("Saving chunks to %s", CHUNKS_PATH)
    with open(CHUNKS_PATH, "wb") as file:
        pickle.dump(chunks, file)

    LOGGER.info("Index build complete")


if __name__ == "__main__":
    main()
