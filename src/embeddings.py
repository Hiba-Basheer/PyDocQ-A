"""
Create sentence embeddings from text chunks and store them in a FAISS index.

This module can be imported for reuse or executed directly for testing
the embedding and indexing pipeline.
"""

import logging
from typing import List

import faiss
from sentence_transformers import SentenceTransformer

from load_and_chunk import chunk_documents, load_txt_documents

# Logging configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Embedding & indexing functions

def create_embeddings(chunks: List) -> "np.ndarray":
    """
    Convert text chunks into vector embeddings.

    Args:
        chunks (List): List of document chunks containing `page_content`.

    Returns:
        np.ndarray: Array of shape (num_chunks, embedding_dimension).
    """
    LOGGER.info("Loading sentence transformer model")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    LOGGER.info("Encoding %d text chunks into embeddings", len(chunks))
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    LOGGER.info("Embeddings created with shape %s", embeddings.shape)
    return embeddings


def store_embeddings(embeddings: "np.ndarray") -> faiss.Index:
    """
    Store embeddings in a FAISS index.

    Args:
        embeddings (np.ndarray): Embedding matrix of shape
            (num_vectors, embedding_dimension).

    Returns:
        faiss.Index: FAISS index containing the embeddings.
    """
    LOGGER.info("Building FAISS index")
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    LOGGER.info("FAISS index created with %d vectors", index.ntotal)
    return index


# Script entry point

def main() -> None:
    """
    Load documents, chunk them, create embeddings, and store them in FAISS.
    """
    data_dir = "data/python_docs"

    LOGGER.info("Loading documents from %s", data_dir)
    documents = load_txt_documents(data_dir)

    LOGGER.info("Chunking documents")
    chunks = chunk_documents(documents)

    embeddings = create_embeddings(chunks)
    index = store_embeddings(embeddings)

    LOGGER.info(
        "Pipeline complete | Embeddings: %s | Index size: %d",
        embeddings.shape,
        index.ntotal,
    )


if __name__ == "__main__":
    main()
