"""
Streamlit application for semantic search and Q&A over Python documentation.

This app loads a FAISS index and document chunks, performs semantic search
using sentence embeddings, and answers user questions using an LLM.
"""

import logging
import pickle
from typing import List, Tuple

import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Configuration

ARTIFACT_DIR = "artifacts"
INDEX_PATH = f"{ARTIFACT_DIR}/faiss.index"
CHUNKS_PATH = f"{ARTIFACT_DIR}/chunks.pkl"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"

TOP_K = 5
MAX_CONTEXT_CHARS = 2000

# Logging configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Resource loading (cached)

@st.cache_resource
def load_resources() -> Tuple[
    faiss.Index,
    List,
    SentenceTransformer,
    pipeline,
]:
    """
    Load and cache FAISS index, document chunks, embedding model, and LLM.

    Returns:
        Tuple containing FAISS index, document chunks, embedding model,
        and text-generation pipeline.
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

    return index, chunks, embed_model, llm


# Retrieval utilities

def semantic_search(
    query: str,
    index: faiss.Index,
    chunks: List,
    embed_model: SentenceTransformer,
    top_k: int = TOP_K,
) -> List:
    """
    Perform semantic search over the FAISS index.

    Args:
        query (str): User query.
        index (faiss.Index): FAISS index.
        chunks (List): List of document chunks.
        embed_model (SentenceTransformer): Embedding model.
        top_k (int): Number of results to retrieve.

    Returns:
        List: Top-matching document chunks.
    """
    LOGGER.debug("Encoding query for semantic search")
    query_embedding = embed_model.encode([query])

    LOGGER.debug("Searching FAISS index (top_k=%d)", top_k)
    _, indices = index.search(query_embedding, top_k)

    return [chunks[i] for i in indices[0]]


def build_context(chunks: List, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Build a context string from retrieved document chunks.

    Args:
        chunks (List): Retrieved document chunks.
        max_chars (int): Maximum allowed context length.

    Returns:
        str: Concatenated context string.
    """
    context_parts = []
    current_length = 0

    for chunk in chunks:
        content = chunk.page_content
        if current_length + len(content) > max_chars:
            break

        context_parts.append(content)
        current_length += len(content)

    return "\n\n".join(context_parts)


# Streamlit UI

st.set_page_config(
    page_title="PyDocQ&A",
    layout="wide",
)

st.title("🧠 PyDocQ&A")
st.subheader("Ask questions about Python")

index, chunks, embed_model, llm = load_resources()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a question about Python:")

if query:
    LOGGER.info("Received user query")
    with st.spinner("Thinking..."):
        top_chunks = semantic_search(
            query=query,
            index=index,
            chunks=chunks,
            embed_model=embed_model,
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

        LOGGER.info("Generating response from LLM")
        response = llm(prompt)[0]["generated_text"]

        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("PyDocQ&A", response))


# Chat display

for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 PyDocQ&A:** {message}")
