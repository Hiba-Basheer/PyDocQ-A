"""
Run semantic search over a FAISS index and answer questions using an LLM.
"""

# Environment

import os

# Prevent Hugging Face from importing TensorFlow / Flax
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

# Standard Imports

import logging
import pickle
from typing import List

import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Configuration

ARTIFACT_DIR = "artifacts"
INDEX_PATH = os.path.join(ARTIFACT_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(ARTIFACT_DIR, "chunks.pkl")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-small"

TOP_K = 10
MAX_CONTEXT_CHARS = 2000
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 200

DEVICE = "cpu"

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Retrieval Utilities

def semantic_search(
    query: str,
    index: faiss.Index,
    chunks: List[str],
    embed_model: SentenceTransformer,
    top_k: int = TOP_K,
) -> List[str]:
    """
    Retrieve top-k relevant document chunks.
    """
    query_embedding = embed_model.encode([query])
    _, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]


def build_context(chunks: List[str], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Concatenate retrieved chunks into a bounded context window.
    """
    context_parts = []
    current_length = 0

    for chunk in chunks:
        if current_length + len(chunk) > max_chars:
            break

        context_parts.append(chunk)
        current_length += len(chunk)

    return "\n\n".join(context_parts)

# Generation

def generate_answer(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
) -> str:
    """
    Generate an answer using the LLM.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main

def main() -> None:
    LOGGER.info("Loading FAISS index")
    index = faiss.read_index(INDEX_PATH)

    LOGGER.info("Loading document chunks")
    with open(CHUNKS_PATH, "rb") as f:
        chunks: List[str] = pickle.load(f)

    LOGGER.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    LOGGER.info("Loading tokenizer and LLM: %s", LLM_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)

    model.to(DEVICE)
    model.eval()

    LOGGER.info("System ready. Type 'exit' to quit.")

    while True:
        query = input("\nAsk a question (or 'exit'): ").strip()
        if query.lower() == "exit":
            LOGGER.info("Exiting application")
            break

        retrieved_chunks = semantic_search(
            query=query,
            index=index,
            chunks=chunks,
            embed_model=embed_model,
        )

        context = build_context(retrieved_chunks)

        prompt = (
            "Answer the question using the context below.\n"
            "If the answer is not in the context, say \"I don't know\".\n\n"
            "Context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{query}"
        )

        answer = generate_answer(prompt, tokenizer, model)
        print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()
