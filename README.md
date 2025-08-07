# RAG-based Document QA System using LangChain and HuggingFace

This project implements a Retrieval-Augmented Generation (RAG) pipeline that allows users to ask questions about a document and receive intelligent answers using local embeddings and Hugging Face models.

## 🚀 Features

- Loads and splits a local text file into manageable chunks
- Creates embeddings using `all-MiniLM-L6-v2`
- Stores and retrieves embeddings via FAISS vector store
- Integrates with Hugging Face models (e.g., Zephyr-7B, Flan-T5)
- Builds a full RAG chain using LangChain



