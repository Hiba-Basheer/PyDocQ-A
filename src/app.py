"""
Streamlit application for semantic search and Q&A over Python documentation.

The app:
- Loads a FAISS index and text chunks
- Retrieves relevant chunks via semantic search
- Uses a Seq2Seq LLM to answer questions based on retrieved context
"""

# Environment

import os

# Prevent Hugging Face from importing TensorFlow / Flax
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

# Standard Imports

import logging
import pickle
from typing import List, Tuple

import faiss
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Configuration

ARTIFACT_DIR = "artifacts"
INDEX_PATH = os.path.join(ARTIFACT_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(ARTIFACT_DIR, "chunks.pkl")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-small"

TOP_K = 4                     
MAX_CONTEXT_CHARS = 1500      
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 200

DEVICE = "cpu"

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Resource Loading 

@st.cache_resource(show_spinner="Loading models and index...")
def load_resources() -> Tuple[
    faiss.Index,
    List[str],
    SentenceTransformer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
]:
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

    return index, chunks, embed_model, tokenizer, model

# Retrieval

def semantic_search(
    query: str,
    index: faiss.Index,
    chunks: List[str],
    embed_model: SentenceTransformer,
    top_k: int = TOP_K,
) -> List[str]:
    query_embedding = embed_model.encode([query])
    _, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]


def build_context(chunks: List[str], max_chars: int = MAX_CONTEXT_CHARS) -> str:
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
            min_new_tokens=40,     # prevents one-word answers
            do_sample=False,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI

st.set_page_config(page_title="PyDocQ&A", layout="wide")
st.title("🧠 PyDocQ&A")
st.subheader("Ask questions about Python documentation")

index, chunks, embed_model, tokenizer, model = load_resources()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a question about Python:")

if query:
    with st.spinner("Thinking..."):
        retrieved_chunks = semantic_search(
            query=query,
            index=index,
            chunks=chunks,
            embed_model=embed_model,
        )

        context = build_context(retrieved_chunks)

        # FLAN-T5 optimized RAG prompt
        prompt = (
            "You are a helpful assistant that answers questions using the given context.\n"
            "If the answer cannot be found in the context, respond with: \"I don't know\".\n\n"
            "Question:\n"
            f"{query}\n\n"
            "Context:\n"
            f"{context}\n\n"
            "Answer:"
        )

        answer = generate_answer(prompt, tokenizer, model)

        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("PyDocQ&A", answer))

# Chat Display

for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 PyDocQ&A:** {message}")
