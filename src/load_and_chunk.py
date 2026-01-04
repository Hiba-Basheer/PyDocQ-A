"""
Load text documents from disk and split them into overlapping chunks.

This module provides lightweight utilities for loading `.txt` files
and chunking them for downstream embedding and retrieval tasks.
"""

import logging
import os
from typing import List

# Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


# Document loading

def load_txt_documents(data_dir: str) -> List[str]:
    """
    Load all `.txt` files from the given directory.

    Args:
        data_dir (str): Path to directory containing text files.

    Returns:
        List[str]: List of document texts.
    """
    LOGGER.info("Loading .txt documents from %s", data_dir)
    documents: List[str] = []

    for filename in os.listdir(data_dir):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(data_dir, filename)
        LOGGER.debug("Loading file: %s", file_path)

        with open(file_path, "r", encoding="utf-8") as file:
            documents.append(file.read())

    LOGGER.info("Loaded %d documents", len(documents))
    return documents


# Document chunking

def chunk_documents(documents: List[str]) -> List[str]:
    """
    Split documents into overlapping text chunks.

    Args:
        documents (List[str]): List of document texts.

    Returns:
        List[str]: List of chunked text segments.
    """
    LOGGER.info("Chunking %d documents", len(documents))
    chunks: List[str] = []

    for doc in documents:
        start = 0
        doc_length = len(doc)

        while start < doc_length:
            end = start + CHUNK_SIZE
            chunk = doc[start:end]
            chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP

    LOGGER.info("Created %d chunks", len(chunks))
    return chunks


# Script entry point

def main() -> None:
    """
    Load documents, chunk them, and print a sample chunk for inspection.
    """
    data_dir = "data/python_docs"

    documents = load_txt_documents(data_dir)
    chunks = chunk_documents(documents)

    if not chunks:
        LOGGER.warning("No chunks created")
        return

    LOGGER.info("Sample Chunk:")
    LOGGER.info(chunks[0][:500])


if __name__ == "__main__":
    main()
