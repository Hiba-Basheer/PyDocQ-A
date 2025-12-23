# PyDocQ&A 🧠  
**An Optimized RAG System for Python Documentation**

PyDocQ&A is a Retrieval-Augmented Generation (RAG) application that enables fast, accurate question answering over Python documentation using semantic search and a transformer-based language model.  
The system is designed with a clear separation between **offline indexing** and **online querying** to ensure low-latency responses.

---

## 🚀 Features

- Semantic search over Python documentation using **FAISS**
- Efficient text embeddings via **Sentence Transformers**
- Transformer-based answer generation using **FLAN-T5**
- One-time offline embedding and indexing for performance optimization
- Interactive **Streamlit** chat interface
- Modular, production-style project structure

---

## 🏗️ Architecture Overview

```
Python Docs (.txt)
↓
Document Chunking
↓
Sentence Embeddings
↓
FAISS Vector Index (Persisted)
↓
Query Embedding
↓
Top-K Semantic Retrieval
↓
Context Injection
↓
LLM Answer Generation
```
---

## 📂 Project Structure

```
PyDocQ&A/
│
├── artifacts/
│ ├── faiss.index # FAISS vector index
│ └── chunks.pkl # Chunked document metadata
│
├── src/
│ ├── load_and_chunk.py # Load and chunk text documents
│ ├── embeddings.py # Create sentence embeddings
│ ├── build_index.py # One-time FAISS index builder
│ ├── qa_chain.py # CLI-based question answering
│ └── app.py # Streamlit UI application
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

- **Python**
- **FAISS** – Vector similarity search
- **SentenceTransformers** – Text embeddings
- **Transformers (FLAN-T5)** – Answer generation
- **Streamlit** – Web UI
- **LangChain** – Document and text utilities

---

📌 Data & Artifacts Note

Raw Python documentation files are excluded from the repository to keep it lightweight and reproducible.

Precomputed FAISS indexes and processed artifacts are included to enable instant evaluation and fast querying without rebuilding embeddings.

📈 Performance Optimization
Stage	Naive Approach	Optimized Approach
Embedding	Every run	One-time offline
Indexing	Every run	Persisted FAISS
Query latency	Minutes	~1–2 seconds
Startup time	High	Low
