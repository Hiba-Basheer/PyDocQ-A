"""
Run semantic search over a FAISS index and answer questions using an LLM.
"""

import os
import logging
import pickle
from typing import List

import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Environment

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

# Configuration

ARTIFACT_DIR = "artifacts"
INDEX_PATH = f"{ARTIFACT_DIR}/faiss.index"
CHUNKS_PATH = f"{ARTIFACT_DIR}/chunks.pkl"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-small"  # faster

TOP_K = 10
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 200

DEVICE = "cpu"

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Utilities

def build_context(chunks, max_chars=2000):
    """
    Build a context string from retrieved document chunks.
    Args:
        chunks (List[str]): List of text chunks (plain strings).
        max_chars (int): Maximum allowed context length.
    Returns:
        str: Concatenated context string.
    """
    context_parts = []
    current_length = 0
    for text in chunks:  
        if current_length + len(text) > max_chars:
            break
        context_parts.append(text)
        current_length += len(text)
    return "\n\n".join(context_parts)



def semantic_search(
    query: str,
    index: faiss.Index,
    chunks: List,
    embed_model: SentenceTransformer,
    top_k: int = TOP_K,
) -> List:
    query_embedding = embed_model.encode([query])
    _, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]


def generate_answer(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
) -> str:
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
        chunks = pickle.load(f)

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
            break

        top_chunks = semantic_search(query, index, chunks, embed_model)
        context = build_context(top_chunks)

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
