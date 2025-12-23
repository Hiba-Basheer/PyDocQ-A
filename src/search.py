"""
Run a semantic search query over in-memory document embeddings.

This script loads text documents, chunks them, creates embeddings,
builds a FAISS index, and executes a test semantic search query.
"""

import logging
from typing import List

import faiss
from sentence_transformers import SentenceTransformer

from embeddings import create_embeddings, store_embeddings
from load_and_chunk import chunk_documents, load_txt_documents

# Logging configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Semantic search

def semantic_search(
    query: str,
    index: faiss.Index,
    chunks: List,
    model: SentenceTransformer,
    top_k: int = 5,
) -> List:
    """
    Search for the most relevant document chunks given a query.

    Args:
        query (str): Search query.
        index (faiss.Index): FAISS index containing embeddings.
        chunks (List): List of document chunks.
        model (SentenceTransformer): Embedding model.
        top_k (int): Number of results to retrieve.

    Returns:
        List: List of top-matching document chunks.
    """
    LOGGER.info("Encoding query for semantic search")
    query_embedding = model.encode([query])

    LOGGER.info("Searching FAISS index (top_k=%d)", top_k)
    _, indices = index.search(query_embedding, top_k)

    return [chunks[idx] for idx in indices[0]]


# Script entry point

def main() -> None:
    """
    Load data, build embeddings, and run a test semantic search query.
    """
    data_dir = "data/python_docs"

    LOGGER.info("Loading documents from %s", data_dir)
    documents = load_txt_documents(data_dir)

    LOGGER.info("Chunking documents")
    chunks = chunk_documents(documents)

    LOGGER.info("Loading embedding model")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    LOGGER.info("Creating embeddings and FAISS index")
    embeddings = create_embeddings(chunks)
    index = store_embeddings(embeddings)

    query = "What is a list comprehension in Python?"
    LOGGER.info("Running test query: %s", query)

    results = semantic_search(
        query=query,
        index=index,
        chunks=chunks,
        model=model,
    )

    print("\nTop results:\n")
    for i, chunk in enumerate(results, start=1):
        print(f"Result {i}")
        print(chunk.page_content[:500])
        print(f"Source: {chunk.metadata.get('source')}\n")


if __name__ == "__main__":
    main()
