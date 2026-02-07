**Notes:**

On some platforms (Windows) faiss-cpu may be difficult to pip-install; use conda or platform wheels if needed: conda install -c conda-forge faiss-cpu
If you get large download times for transformers/torch, consider using smaller models (e.g., t5-small) for quick tests.
Quick test run (small-scale) — useful to verify the pipeline without heavy downloads
Use reduced numbers so this runs quickly: python hybrid_rag_eval.py build_index --generate_fixed --fixed_n 10 --random_n 20 --min_words 50 --chunks_out data/test_chunks.jsonl --index_dense data/test_index.faiss --index_bm25 data/test_bm25
This:

Generates a small fixed set (10 pages) and a random set (20 pages)

Builds chunk file data/test_chunks.jsonl and the two indexes

**Generate a small evaluation question set:**

python hybrid_rag_eval.py generate_questions --chunks_in data/test_chunks.jsonl --out_questions data/test_questions.jsonl --num_questions 10

**Run evaluation (use a small generator model to reduce downloads):**

python hybrid_rag_eval.py run_eval --chunks_in data/test_chunks.jsonl --index_dense data/test_index.faiss --index_bm25 data/test_bm25 --questions data/test_questions.jsonl --out_prefix results/test_report --gen_model t5-small --device cpu --top_n 5 --recall_k 5

**Full pipeline (assignment-scale)**
**Generate fixed set once (per group). This is time-consuming and requires internet. Run once and commit fixed_urls.json to your group repo**

python hybrid_rag_eval.py build_index --generate_fixed --fixed_n 200 --random_n 300 --min_words 200 --chunks_out data/processed/chunks.jsonl --index_dense data/index.faiss --index_bm25 data/bm25_index

**Or if you already have fixed_urls.json available:**

python hybrid_rag_eval.py build_index --fixed fixed_urls.json --random_n 300 --min_words 200 --chunks_out data/processed/chunks.jsonl --index_dense data/index.faiss --index_bm25 data/bm25_index

**Generate 100 evaluation Qs:**

python hybrid_rag_eval.py generate_questions --chunks_in data/processed/chunks.jsonl --out_questions data/test_questions.jsonl --num_questions 100

**Run full evaluation (this will download and use the generator model — may be slow):**

python hybrid_rag_eval.py run_eval --chunks_in data/processed/chunks.jsonl --index_dense data/index.faiss --index_bm25 data/bm25_index --questions data/test_questions.jsonl --out_prefix results/report --gen_model google/flan-t5-base --device cpu --top_n 10 --recall_k 10

**Run query_rag:**

python query_rag.py

**Options you may change:**

--gen_model: choose a smaller model (e.g., t5-small) for speed during development.
--device: set to "cuda" if you have a GPU and proper CUDA-enabled PyTorch.
--top_n / --recall_k / --random_n etc. control retrieval and sampling sizes.
Running the Streamlit UI

***Start the Streamlit app:***

streamlit run hybrid_rag_eval.py

**In the UI sidebar set:**

Chunks JSONL: path to your chunks file (e.g., data/processed/chunks.jsonl)
Dense index: data/index.faiss
BM25 artifacts prefix: data/bm25_index
Embedding & generator models (embedding default all-MiniLM-L6-v2)
Click "Load Indexes & Models", then type a query and press "Retrieve & Generate". Notes:

First-time model loading can take several minutes as HF models are downloaded.

For very large indexes/models, loading in the Streamlit session may use a lot of memory.

**Troubleshooting & tips**

Wikipedia sampling:
get_random_wikipedia_url uses the wikipedia package and requires internet; expect some retries due to disambiguation pages. Respect Wikimedia rate limits and cache the fixed set once created.
FAISS install issues:
If pip install faiss-cpu fails on your OS, try conda: conda install -c conda-forge faiss-cpu
Speed:
Embedding 500 pages and building FAISS can take time. Consider using batching or offline embedding cache for repeated runs.
CI / offline tests:
Add a lightweight test set and mock model calls to run fast in CI (see earlier suggestions).
Disk & cache:
Hugging Face caches models in ~/.cache/huggingface/hub — set HF_HOME or HUGGINGFACE_HUB_CACHE env if needed.
Example end-to-end commands (full-size)
One-time fixed set generation: python hybrid_rag_eval.py build_index --generate_fixed --fixed_n 200 --random_n 300 --min_words 200
Generate Qs: python hybrid_rag_eval.py generate_questions --chunks_in data/processed/chunks.jsonl --out_questions data/eval/questions.jsonl --num_questions 100
Evaluate: python hybrid_rag_eval.py run_eval --chunks_in data/processed/chunks.jsonl --index_dense data/index.faiss --index_bm25 data/bm25_index --questions data/eval/questions.jsonl --out_prefix results/report --gen_model google/flan-t5-base --device cpu --top_n 10 --recall_k 10

**Pre-requisites for streamlit:**
https://docs.streamlit.io/get-started/installation/community-cloud
https://share.streamlit.io/
https://share.streamlit.io/