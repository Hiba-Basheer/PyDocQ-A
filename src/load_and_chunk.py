"""
Load text documents from disk and split them into semantically
meaningful, overlapping chunks.

This module provides lightweight utilities for loading `.txt` files
and chunking them for downstream embedding and retrieval tasks,
while preserving paragraph-level context.
"""

import logging
import os
import re
from typing import List

# =========================
# Configuration
# =========================

CHUNK_SIZE = 500          # Maximum characters per chunk
CHUNK_OVERLAP = 100       # Overlap between consecutive chunks

# =========================
# Logging configuration
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# =========================
# Document loading
# =========================

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

# =========================
# Document chunking
# =========================

def chunk_documents(documents: List[str]) -> List[str]:
    """
    Split documents into semantically meaningful, overlapping chunks.

    Chunking is performed at paragraph boundaries to avoid
    splitting sentences, definitions, or code examples.

    Args:
        documents (List[str]): List of document texts.

    Returns:
        List[str]: List of chunked text segments.
    """
    LOGGER.info("Chunking %d documents", len(documents))
    chunks: List[str] = []

    for doc in documents:
        # Normalize excessive newlines
        doc = re.sub(r"\n{2,}", "\n\n", doc.strip())

        paragraphs = doc.split("\n\n")
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(paragraph) > CHUNK_SIZE:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                    # Create overlap from the end of the previous chunk
                    current_chunk = current_chunk[-CHUNK_OVERLAP:]
                else:
                    # Paragraph itself is too large
                    chunks.append(paragraph[:CHUNK_SIZE])
                    continue

            current_chunk += "\n\n" + paragraph

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

    LOGGER.info("Created %d chunks", len(chunks))
    return chunks

# =========================
# Script entry point
# =========================

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
