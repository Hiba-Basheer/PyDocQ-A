"""
Run semantic search over a FAISS index and answer questions using an LLM.

This script loads a persisted FAISS index and document chunks, performs
semantic search using sentence embeddings, builds a context window, and
generates answers using a text-to-text transformer model.
"""

import logging
import pickle
from typing import List

import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Configuration

ARTIFACT_DIR = "artifacts"
INDEX_PATH = f"{ARTIFACT_DIR}/faiss.index"
CHUNKS_PATH = f"{ARTIFACT_DIR}/chunks.pkl"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"

MAX_CONTEXT_CHARS = 2000
TOP_K = 5

# Logging configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Retrieval utilities

def build_context(chunks: List, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Build a context string from retrieved document chunks.

    Args:
        chunks (List): List of LangChain Document chunks.
        max_chars (int): Maximum number of characters allowed in the context.

    Returns:
        str: Concatenated context string.
    """
    context = []

    current_length = 0
    for chunk in chunks:
        content = chunk.page_content
        if current_length + len(content) > max_chars:
            break

        context.append(content)
        current_length += len(content)

    return "\n\n".join(context)


def semantic_search(
    query: str,
    index: faiss.Index,
    chunks: List,
    model: SentenceTransformer,
    top_k: int = TOP_K,
) -> List:
    """
    Perform semantic search over a FAISS index.

    Args:
        query (str): User query.
        index (faiss.Index): FAISS index containing embeddings.
        chunks (List): List of document chunks.
        model (SentenceTransformer): Embedding model.
        top_k (int): Number of top results to retrieve.

    Returns:
        List: List of top-matching document chunks.
    """
    LOGGER.debug("Encoding query for semantic search")
    query_embedding = model.encode([query])

    LOGGER.debug("Searching FAISS index (top_k=%d)", top_k)
    _, indices = index.search(query_embedding, top_k)

    return [chunks[i] for i in indices[0]]


# Script entry point

def main() -> None:
    """
    Run an interactive question-answering loop over indexed documents.
    """
    LOGGER.info("Loading FAISS index from %s", INDEX_PATH)
    index = faiss.read_index(INDEX_PATH)

    LOGGER.info("Loading document chunks from %s", CHUNKS_PATH)
    with open(CHUNKS_PATH, "rb") as file:
        chunks = pickle.load(file)

    LOGGER.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    LOGGER.info("Loading language model: %s", LLM_MODEL_NAME)
    llm = pipeline(
        task="text2text-generation",
        model=LLM_MODEL_NAME,
        max_length=300,
    )

    LOGGER.info("System ready. Enter 'exit' to quit.")

    while True:
        query = input("\nAsk a question (or 'exit'): ").strip()
        if query.lower() == "exit":
            LOGGER.info("Exiting application")
            break

        LOGGER.info("Running semantic search")
        top_chunks = semantic_search(
            query=query,
            index=index,
            chunks=chunks,
            model=embed_model,
        )

        context = build_context(top_chunks)

        prompt = (
            "Answer the question using the context below.\n"
            "If the answer is not in the context, say \"I don't know\".\n\n"
            "Context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{query}"
        )

        LOGGER.info("Generating answer")
        response = llm(prompt)

        print("\nAnswer:\n", response[0]["generated_text"])


if __name__ == "__main__":
    main()
