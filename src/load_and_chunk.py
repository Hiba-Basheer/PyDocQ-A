"""
Load text documents from disk and split them into overlapping chunks.

This module provides utilities for loading `.txt` files using LangChain
and chunking them for downstream embedding and retrieval tasks.
"""

import logging
import os
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Logging configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Document loading

def load_txt_documents(data_dir: str) -> List:
    """
    Load all `.txt` files from the given directory.

    Args:
        data_dir (str): Path to directory containing text files.

    Returns:
        List: List of LangChain Document objects.
    """
    LOGGER.info("Loading .txt documents from %s", data_dir)
    documents = []

    for filename in os.listdir(data_dir):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(data_dir, filename)
        LOGGER.debug("Loading file: %s", file_path)

        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        documents.extend(docs)

    LOGGER.info("Loaded %d documents", len(documents))
    return documents


# Document chunking

def chunk_documents(documents: List) -> List:
    """
    Split documents into overlapping text chunks.

    Args:
        documents (List): List of LangChain Document objects.

    Returns:
        List: List of chunked Document objects.
    """
    LOGGER.info("Chunking %d documents", len(documents))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

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

    sample = chunks[0]
    LOGGER.info("--- Sample Chunk ---")
    LOGGER.info(sample.page_content[:500])
    LOGGER.info("Source: %s", sample.metadata.get("source"))


if __name__ == "__main__":
    main()
