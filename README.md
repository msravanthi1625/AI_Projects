# My First RAG Project

This repository demonstrates a minimal Retrieval-Augmented Generation (RAG) pipeline:
- Prepare a small document corpus
- Chunk & embed documents with sentence-transformers
- Index vectors in FAISS
- Retrieve top-k contexts for a query
- Generate an answer using a seq2seq model (or an LLM)

Quickstart
1. Create venv and install:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Prepare data in `data/raw/`, run `src/embed_index.py` to build index.

3. Run `src/query_rag.py` to try queries against the local index.

See `notebooks/quickstart.ipynb` for an interactive demo.

License: MIT