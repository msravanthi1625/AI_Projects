"""
hybrid_rag_eval.py

Sample scaffold for:
Assignment 2 – Hybrid RAG System with Automated Evaluation
(with Streamlit UI stub wired to Retriever + Generator)

This updated version integrates automated Wikipedia sampling helpers:
- get_random_wikipedia_url(min_words=200)
- generate_fixed_set(n=200, min_words=200)
- generate_random_set(n=300, min_words=200, exclude_titles=None)

Notes:
- Requires the `wikipedia` Python package (pip install wikipedia).
- The build_index command will now:
    - if a fixed_urls.json file exists, load it (expects list of objects with url/title/text)
    - otherwise, if --generate_fixed is provided, create fixed_urls.json via generate_fixed_set
    - always generate the random 300-page set from live Wikipedia (excluding fixed titles)
- Make sure to respect Wikimedia usage policies and rate limits. Consider caching results.

To run Streamlit UI:
    streamlit run hybrid_rag_eval.py
"""
import argparse
import json
import os
import random
import time
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# third-party ML libs
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# new dependency for Wikipedia helpers
try:
    import wikipedia
except Exception as e:
    wikipedia = None
    # We'll raise at runtime if user attempts to use wiki functions without the package.

# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Chunk:
    id: str
    url: str
    title: str
    text: str
    token_count: int


@dataclass
class QAItem:
    qid: str
    question: str
    answer: str
    answer_url: str  # ground-truth source URL (URL-level eval)
    category: str = "factual"  # e.g., factual/comparative/multi-hop/inferential


# ----------------------------
# Wikipedia helpers (new)
# ----------------------------
def get_random_wikipedia_url(min_words: int = 200) -> Dict[str, str]:
    """
    Sample a random Wikipedia page title, fetch the page, and return
    a dict {title, url, text} where text has at least min_words words.

    Uses the `wikipedia` package (https://pypi.org/project/wikipedia/).
    Retries on DisambiguationError and PageError.
    """
    if wikipedia is None:
        raise ImportError("The 'wikipedia' package is required for get_random_wikipedia_url. Install with `pip install wikipedia`.")
    while True:
        title = wikipedia.random()
        try:
            page = wikipedia.page(title)
            text = page.content
            if len(text.split()) >= min_words:
                return {
                    "title": page.title,
                    "url": page.url,
                    "text": text
                }
        except (wikipedia.DisambiguationError, wikipedia.PageError):
            continue
        except Exception:
            # network/other transient error: sleep a little and retry
            time.sleep(1)
            continue


def generate_fixed_set(n: int = 200, min_words: int = 200, out_path: str = "fixed_urls.json") -> List[Dict[str, str]]:
    """
    Generate a fixed set of n unique Wikipedia pages (title, url, text) and write to out_path.
    This should be run once per group to create the assignment's fixed set.
    """
    if wikipedia is None:
        raise ImportError("The 'wikipedia' package is required for generate_fixed_set. Install with `pip install wikipedia`.")
    urls = []
    titles = set()
    print(f"Generating fixed set of {n} pages (min {min_words} words)...")
    while len(urls) < n:
        page = get_random_wikipedia_url(min_words)
        if page["title"] not in titles:
            urls.append({"url": page["url"], "title": page["title"], "text": page["text"]})
            titles.add(page["title"])
            if len(urls) % 10 == 0:
                print(f"  collected {len(urls)}/{n}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(urls, f, indent=2, ensure_ascii=False)
    print(f"Wrote fixed set to {out_path}")
    return urls


def generate_random_set(n: int = 300, min_words: int = 200, exclude_titles: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Generate n unique random Wikipedia pages, excluding any titles in exclude_titles.
    Returns a list of dicts {url, title, text}. Does NOT write to disk by default.
    """
    if wikipedia is None:
        raise ImportError("The 'wikipedia' package is required for generate_random_set. Install with `pip install wikipedia`.")
    urls = []
    titles = set(exclude_titles) if exclude_titles else set()
    print(f"Generating random set of {n} pages (min {min_words} words), excluding {len(titles)} titles...")
    while len(urls) < n:
        page = get_random_wikipedia_url(min_words)
        if page["title"] not in titles:
            urls.append({"url": page["url"], "title": page["title"], "text": page["text"]})
            titles.add(page["title"])
            if len(urls) % 50 == 0:
                print(f"  collected {len(urls)}/{n}")
    print(f"Generated random set of {n} pages.")
    return urls


# ----------------------------
# Chunking utility
# ----------------------------
def chunk_text(text: str, url: str, title: str, chunk_size_tokens: int = 300, overlap_tokens: int = 50) -> List[Chunk]:
    """
    Very simple whitespace-based chunking as a placeholder.
    Replace with tokenizer-based chunking to control token counts (e.g., HF tokenizer).
    """
    words = text.split()
    chunks = []
    i = 0
    chunk_id_base = abs(hash(url)) % (10 ** 8)
    idx = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size_tokens]
        chunk_text_str = " ".join(chunk_words)
        chunk = Chunk(
            id=f"{chunk_id_base}_{idx}",
            url=url,
            title=title,
            text=chunk_text_str,
            token_count=len(chunk_words)
        )
        chunks.append(chunk)
        idx += 1
        i += chunk_size_tokens - overlap_tokens
    return chunks


# ----------------------------
# Corpus Builder (updated to use generated fixed/random sets)
# ----------------------------
def build_corpus_from_wikipedia(fixed_list: List[Dict[str, str]], random_list: List[Dict[str, str]], output_path: str, min_words: int = 200):
    """
    Build corpus from two lists of page dicts:
      fixed_list: list of {"url","title","text"} (200 pages)
      random_list: list of {"url","title","text"} (300 pages)
    Chunk and write to JSONL (one chunk per line).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_pages = fixed_list + random_list
    chunks_out = []
    for page in all_pages:
        url = page["url"]
        title = page.get("title", "")
        text = page.get("text", "")
        if len(text.split()) < min_words:
            # skip if unexpectedly short
            print(f"Skipping {url} due to short length ({len(text.split())} words)")
            continue
        chunks = chunk_text(text, url, title, chunk_size_tokens=300, overlap_tokens=50)
        for c in chunks:
            chunks_out.append(asdict(c))
    with open(output_path, "w", encoding="utf-8") as f:
        for c in chunks_out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {len(chunks_out)} chunks to {output_path}")


# ----------------------------
# Dense Indexer (SentenceTransformer + FAISS)
# ----------------------------
class DenseIndexer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embed_model = SentenceTransformer(model_name)
        self.index = None
        self.id_map: List[str] = []

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    def build_index(self, chunks_jsonl_path: str, index_path: str):
        texts = []
        ids = []
        with open(chunks_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                ids.append(obj["id"])
                texts.append(obj["text"])
        embeddings = self.embed_texts(texts)
        # normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)
        # save id_map
        with open(index_path + ".ids.json", "w", encoding="utf-8") as f:
            json.dump(ids, f)
        print(f"Saved dense index to {index_path} and ids to {index_path}.ids.json")

    def load_index(self, index_path: str):
        self.index = faiss.read_index(index_path)
        with open(index_path + ".ids.json", "r", encoding="utf-8") as f:
            self.id_map = json.load(f)

    def search(self, query: str, k: int = 100) -> List[Tuple[str, float]]:
        q_emb = self.embed_texts([query])
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, k)
        results = []
        for rank, (idx, score) in enumerate(zip(I[0], D[0])):
            if idx < 0 or idx >= len(self.id_map):
                continue
            results.append((self.id_map[idx], float(score)))
        return results


# ----------------------------
# Sparse Indexer (BM25)
# ----------------------------
class BM25Indexer:
    def __init__(self):
        self.bm25 = None
        self.id_map: List[str] = []
        self.tokenized_docs: List[List[str]] = []

    def build(self, chunks_jsonl_path: str, index_path: str):
        docs = []
        ids = []
        tokenized = []
        with open(chunks_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                docs.append(obj["text"])
                ids.append(obj["id"])
        # very simple whitespace tokenization; replace with better tokenizer if needed
        tokenized = [d.split() for d in docs]
        self.bm25 = BM25Okapi(tokenized)
        self.id_map = ids
        self.tokenized_docs = tokenized
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path + ".ids.json", "w", encoding="utf-8") as f:
            json.dump(ids, f)
        # persist docs for reload
        with open(index_path + ".docs.json", "w", encoding="utf-8") as f:
            json.dump(docs, f)
        print(f"Saved BM25 artifacts to {index_path}.ids.json and {index_path}.docs.json")

    def load(self, index_path: str):
        with open(index_path + ".ids.json", "r", encoding="utf-8") as f:
            self.id_map = json.load(f)
        with open(index_path + ".docs.json", "r", encoding="utf-8") as f:
            docs = json.load(f)
        self.tokenized_docs = [d.split() for d in docs]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, k: int = 100) -> List[Tuple[str, float]]:
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)  # array over docs
        topk_idx = np.argsort(scores)[::-1][:k]
        results = [(self.id_map[i], float(scores[i])) for i in topk_idx]
        return results


# ----------------------------
# Reciprocal Rank Fusion (RRF)
# ----------------------------
class RRFCombiner:
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def fuse(self, dense_results: List[Tuple[str, float]], sparse_results: List[Tuple[str, float]], top_n: int = 10) -> List[Tuple[str, float]]:
        rank_map: Dict[str, Dict[str, int]] = {}
        for rank, (cid, _) in enumerate(dense_results, start=1):
            rank_map.setdefault(cid, {})["dense"] = rank
        for rank, (cid, _) in enumerate(sparse_results, start=1):
            rank_map.setdefault(cid, {})["sparse"] = rank
        rrf_scores: List[Tuple[str, float]] = []
        for cid, info in rank_map.items():
            score = 0.0
            if "dense" in info:
                score += 1.0 / (self.rrf_k + info["dense"])
            if "sparse" in info:
                score += 1.0 / (self.rrf_k + info["sparse"])
            rrf_scores.append((cid, score))
        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        return rrf_scores[:top_n]


# ----------------------------
# Generator wrapper (HF example)
# ----------------------------
class Generator:
    def __init__(self, model_name: str = "google/flan-t5-base", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def generate(self, query: str, context_chunks: List[str], max_new_tokens: int = 128) -> str:
        context = "\n\n".join(context_chunks)
        prompt = f"Context:{context}  Question: {query} Answer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        ans = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return ans.strip()


# ----------------------------
# Retrieval Orchestrator
# ----------------------------
class Retriever:
    def __init__(self, dense_indexer: DenseIndexer, bm25_indexer: BM25Indexer, chunks_jsonl_path: str):
        self.dense = dense_indexer
        self.bm25 = bm25_indexer
        self.chunks_map = {}  # chunk_id -> chunk dict
        with open(chunks_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.chunks_map[obj["id"]] = obj

    def retrieve(self, query: str, k_dense: int = 100, k_sparse: int = 100, top_n: int = 10, rrf_k: int = 60) -> List[Dict[str, Any]]:
        dense_res = self.dense.search(query, k=k_dense)
        sparse_res = self.bm25.search(query, k=k_sparse)
        combiner = RRFCombiner(rrf_k=rrf_k)
        fused = combiner.fuse(dense_res, sparse_res, top_n=top_n)
        results = []
        dense_ranks = {cid: rank for rank, (cid, _) in enumerate(dense_res, start=1)}
        sparse_ranks = {cid: rank for rank, (cid, _) in enumerate(sparse_res, start=1)}
        for cid, rrf_score in fused:
            chunk = self.chunks_map[cid]
            results.append({
                "id": cid,
                "url": chunk["url"],
                "title": chunk.get("title", ""),
                "text": chunk["text"],
                "rrf_score": rrf_score,
                "dense_rank": dense_ranks.get(cid, None),
                "sparse_rank": sparse_ranks.get(cid, None)
            })
        return results


# ----------------------------
# Evaluation Metrics
# ----------------------------
class Evaluator:
    @staticmethod
    def compute_mrr_url(ground_truth_url: str, retrieved_urls: List[str]) -> float:
        for rank, url in enumerate(retrieved_urls, start=1):
            if url == ground_truth_url:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def exact_match(pred: str, gold: str) -> int:
        def norm(s: str) -> str:
            return " ".join(s.lower().strip().split())
        return int(norm(pred) == norm(gold))

    @staticmethod
    def recall_at_k(ground_truth_url: str, retrieved_urls: List[str], k: int = 10) -> int:
        return int(ground_truth_url in retrieved_urls[:k])

    def evaluate_all(self, qa_items: List[QAItem], retriever: Retriever, generator: Generator,
                     top_n: int = 10, recall_k: int = 10) -> Tuple[pd.DataFrame, Dict[str, float]]:
        rows = []
        mrrs = []
        ems = []
        recalls = []
        times = []
        for qa in qa_items:
            start = time.time()
            retrieved = retriever.retrieve(qa.question, top_n=top_n)
            retrieved_urls = [r["url"] for r in retrieved]
            context_texts = [r["text"] for r in retrieved]
            answer = generator.generate(qa.question, context_texts)
            elapsed = time.time() - start
            mrr_val = self.compute_mrr_url(qa.answer_url, retrieved_urls)
            em_val = self.exact_match(answer, qa.answer)
            recall_val = self.recall_at_k(qa.answer_url, retrieved_urls, k=recall_k)
            rows.append({
                "qid": qa.qid,
                "question": qa.question,
                "gold_answer": qa.answer,
                "gold_url": qa.answer_url,
                "predicted_answer": answer,
                "mrr_url": mrr_val,
                "exact_match": em_val,
                "recall_at_k": recall_val,
                "time": elapsed,
                "retrieved_urls": retrieved_urls
            })
            mrrs.append(mrr_val)
            ems.append(em_val)
            recalls.append(recall_val)
            times.append(elapsed)
        df = pd.DataFrame(rows)
        agg = {
            "MRR_URL": float(np.mean(mrrs)),
            "Exact_Match": float(np.mean(ems)),
            f"Recall@{recall_k}": float(np.mean(recalls)),
            "Avg_Time": float(np.mean(times))
        }
        return df, agg


# ----------------------------
# Question Generator (Automated)
# ----------------------------
class QuestionGenerator:
    @staticmethod
    def generate_from_corpus(chunks_jsonl_path: str, out_questions_path: str, num_questions: int = 100):
        qa_items = []
        with open(chunks_jsonl_path, "r", encoding="utf-8") as f:
            chunks = [json.loads(l) for l in f]
        for i in range(num_questions):
            c = random.choice(chunks)
            sentences = [s.strip() for s in c["text"].split(".") if s.strip()]
            if len(sentences) < 2:
                continue
            question = sentences[0] + "?"
            answer = sentences[1]
            qa = QAItem(qid=f"Q{i}", question=question, answer=answer, answer_url=c["url"], category="factual")
            qa_items.append(qa)
        with open(out_questions_path, "w", encoding="utf-8") as f:
            for qa in qa_items:
                f.write(json.dumps(asdict(qa), ensure_ascii=False) + "\n")
        print(f"Wrote {len(qa_items)} generated questions to {out_questions_path}")


# ----------------------------
# Reporting utilities
# ----------------------------
def export_report(df: pd.DataFrame, agg: Dict[str, float], out_prefix: str):
    csv_path = out_prefix + ".csv"
    df.to_csv(csv_path, index=False)
    with open(out_prefix + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    plt.figure(figsize=(8, 4))
    plt.hist(df["mrr_url"].fillna(0.0), bins=20)
    plt.title("MRR (per-question) distribution")
    plt.xlabel("MRR")
    plt.ylabel("Count")
    plt.savefig(out_prefix + ".mrr_hist.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(df["time"].fillna(0.0), bins=20)
    plt.title("Response time distribution")
    plt.xlabel("Seconds")
    plt.ylabel("Count")
    plt.savefig(out_prefix + ".time_hist.png")
    plt.close()
    print(f"Exported CSV -> {csv_path}, summary JSON, and plots with prefix {out_prefix}")


# ----------------------------
# CLI / Pipeline Orchestration (updated build_index)
# ----------------------------
def cmd_build_index(args):
    """
    Build corpus and indexes.

    Behavior:
      - If --generate_fixed is set or fixed file doesn't exist: generate fixed set and write fixed_urls.json
      - Always generate a fresh random set (300 pages) excluding the fixed titles
      - Build chunks JSONL and indexes
    """
    # load or create fixed set
    fixed_pages = None
    if args.fixed and os.path.exists(args.fixed):
        with open(args.fixed, "r", encoding="utf-8") as f:
            fixed_pages = json.load(f)
        print(f"Loaded fixed set from {args.fixed} ({len(fixed_pages)} pages)")
    elif args.generate_fixed:
        fixed_pages = generate_fixed_set(n=args.fixed_n, min_words=args.min_words, out_path=args.fixed or "fixed_urls.json")
    else:
        raise ValueError("No fixed_urls.json found. Provide --fixed path or use --generate_fixed to create one.")

    fixed_titles = [p["title"] for p in fixed_pages]

    # generate random set (new each run) excluding fixed titles
    random_pages = generate_random_set(n=args.random_n, min_words=args.min_words, exclude_titles=fixed_titles)

    # build corpus from the two lists
    build_corpus_from_wikipedia(fixed_pages, random_pages, args.chunks_out, min_words=args.min_words)

    # Build dense index
    dense = DenseIndexer(model_name=args.embed_model)
    dense.build_index(args.chunks_out, args.index_dense)

    # Build BM25 index
    bm25 = BM25Indexer()
    bm25.build(args.chunks_out, args.index_bm25)


def cmd_generate_questions(args):
    QuestionGenerator.generate_from_corpus(args.chunks_in, args.out_questions, num_questions=args.num_questions)


def cmd_run_eval(args):
    dense = DenseIndexer(model_name=args.embed_model)
    dense.load_index(args.index_dense)
    bm25 = BM25Indexer()
    bm25.load(args.index_bm25)
    retriever = Retriever(dense, bm25, args.chunks_in)
    generator = Generator(model_name=args.gen_model, device=args.device)
    qa_items = []
    with open(args.questions, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qa_items.append(QAItem(**obj))
    evaluator = Evaluator()
    df, agg = evaluator.evaluate_all(qa_items, retriever, generator, top_n=args.top_n, recall_k=args.recall_k)
    export_report(df, agg, args.out_prefix)


def cmd_full_pipeline(args):
    cmd_build_index(args)
    cmd_generate_questions(args)
    args.questions = args.out_questions
    cmd_run_eval(args)


# ----------------------------
# Streamlit UI stub (unchanged)
# ----------------------------
def streamlit_app():
    """
    Streamlit-based interactive UI to query the hybrid RAG pipeline.
    To run: install streamlit and run:
      streamlit run hybrid_rag_eval.py
    The app will:
      - Allow user to load existing indexes and chunk file
      - Enter a query and run retrieve + generate
      - Show generated answer, timing, and top retrieved chunks with ranks/scores and URLs
    Note: Keep models and indexes on disk and point the UI to their paths.
    """
    try:
        import streamlit as st
    except Exception:
        print("Streamlit not installed; skipping Streamlit UI.")
        return

    st.set_page_config(page_title="Hybrid RAG Demo", layout="wide")
    st.title("Hybrid RAG — Demo UI (Streamlit Stub)")

    # Sidebar: index / model configuration
    st.sidebar.header("Configuration")
    chunks_path = st.sidebar.text_input("Chunks JSONL", value="data/processed/chunks.jsonl")
    index_dense = st.sidebar.text_input("Dense index (FAISS)", value="data/index.faiss")
    index_bm25 = st.sidebar.text_input("BM25 artifacts prefix", value="data/bm25_index")
    embed_model = st.sidebar.text_input("Embedding model", value="all-MiniLM-L6-v2")
    gen_model = st.sidebar.text_input("Generator model", value="google/flan-t5-base")
    top_n = st.sidebar.number_input("Top-N (RRF output)", min_value=1, max_value=50, value=10, step=1)
    k_dense = st.sidebar.number_input("k_dense (dense top-K)", min_value=1, max_value=500, value=100, step=1)
    k_sparse = st.sidebar.number_input("k_sparse (sparse top-K)", min_value=1, max_value=500, value=100, step=1)
    rrf_k = st.sidebar.number_input("RRF k", min_value=1, max_value=1000, value=60, step=1)
    device = st.sidebar.selectbox("Device for generator", options=["cpu", "cuda"], index=0)

    if "loaded" not in st.session_state:
        st.session_state["loaded"] = False

    def load_artifacts():
        st.session_state["status"] = "Loading indexes and models..."
        try:
            dense_idx = DenseIndexer(model_name=embed_model)
            dense_idx.load_index(index_dense)
            bm25_idx = BM25Indexer()
            bm25_idx.load(index_bm25)
            retriever = Retriever(dense_idx, bm25_idx, chunks_path)
            generator = Generator(model_name=gen_model, device=device)
            st.session_state["dense_idx"] = dense_idx
            st.session_state["bm25_idx"] = bm25_idx
            st.session_state["retriever"] = retriever
            st.session_state["generator"] = generator
            st.session_state["status"] = "Loaded indexes and models."
            st.session_state["loaded"] = True
        except Exception as e:
            st.session_state["status"] = f"Error loading artifacts: {e}"
            st.session_state["loaded"] = False

    if st.sidebar.button("Load Indexes & Models"):
        load_artifacts()

    st.sidebar.markdown("---")
    st.sidebar.write(st.session_state.get("status", "Not loaded"))

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Query")
        query = st.text_area("Enter your question here", value="", height=120)
        run_button = st.button("Retrieve & Generate")
    with col2:
        st.subheader("Options / Diagnostics")
        st.write("Top-N:", top_n)
        st.write("k_dense:", k_dense, "k_sparse:", k_sparse, "RRF k:", rrf_k)

    if run_button:
        if not st.session_state.get("loaded", False):
            st.error("Indexes and models are not loaded. Click 'Load Indexes & Models' first.")
        elif not query.strip():
            st.warning("Please enter a query.")
        else:
            retriever: Retriever = st.session_state["retriever"]
            generator: Generator = st.session_state["generator"]
            start = time.time()
            retrieved = retriever.retrieve(query, k_dense, k_sparse, top_n=top_n, rrf_k=rrf_k)
            retrieved_urls = [r["url"] for r in retrieved]
            context_texts = [r["text"] for r in retrieved]
            gen_start = time.time()
            answer = generator.generate(query, context_texts)
            gen_time = time.time() - gen_start
            total_time = time.time() - start

            st.markdown("### Generated Answer")
            st.info(answer)
            st.write(f"Generation time: {gen_time:.2f}s  —  Total RT: {total_time:.2f}s")

            st.markdown("### Retrieved Chunks (RRF fused)")
            table_rows = []
            for i, r in enumerate(retrieved, start=1):
                excerpt = r["text"][:400].replace("\n", " ")
                table_rows.append({
                    "rank": i,
                    "rrf_score": r["rrf_score"],
                    "dense_rank": r["dense_rank"],
                    "sparse_rank": r["sparse_rank"],
                    "url": r["url"],
                    "title": r["title"],
                    "excerpt": excerpt
                })
            df_view = pd.DataFrame(table_rows)
            st.dataframe(df_view[["rank", "rrf_score", "dense_rank", "sparse_rank", "title", "url", "excerpt"]])

            for i, r in enumerate(retrieved, start=1):
                with st.expander(f"Chunk {i} — {r['title']} — {r['url']}"):
                    st.write(f"RRF score: {r['rrf_score']}, dense_rank: {r['dense_rank']}, sparse_rank: {r['sparse_rank']}")
                    st.write(r["text"])

            if st.button("Show assembled prompt"):
                joined_context = "\n\n".join(context_texts)
                prompt = f"Context:\n\n{joined_context}\n\nQuestion: {query}\nAnswer:"
                st.code(prompt)

    st.markdown("---")
    st.write("Tip: Use the sidebar to point to local index files and models. Loading large models may take time.")


# If streamlit is available when this file is executed (via `streamlit run`),
# run the app. Guard import to avoid requiring streamlit for CLI usage.
if __name__ != "__main__":
    # If imported as a module, do nothing on import.
    pass
else:
    # When executed as a script, check if streamlit is running it (streamlit sets certain env)
    if "STREAMLIT_RUN" in os.environ or any("streamlit" in x for x in sys.argv):
        # Call streamlit_app if streamlit run invoked the script
        try:
            streamlit_app()
        except Exception as e:
            print(f"Streamlit app failed to start: {e}")

    # CLI entrypoint
    def main():
        parser = argparse.ArgumentParser(description="Hybrid RAG System + Automated Evaluation scaffold")
        sub = parser.add_subparsers(dest="cmd")
        # build_index
        p_build = sub.add_parser("build_index")
        p_build.add_argument("--fixed", default="fixed_urls.json", help="JSON file with 200 fixed pages (objects with url,title,text).")
        p_build.add_argument("--generate_fixed", action="store_true", help="Generate fixed_urls.json by sampling Wikipedia (run once per group).")
        p_build.add_argument("--fixed_n", type=int, default=200, help="Number of fixed pages to generate if --generate_fixed used.")
        p_build.add_argument("--random_n", type=int, default=300, help="Number of random pages to sample for this run.")
        p_build.add_argument("--min_words", type=int, default=200, help="Minimum words per Wikipedia page.")
        p_build.add_argument("--chunks_out", default="data/processed/chunks.jsonl")
        p_build.add_argument("--index_dense", default="data/index.faiss")
        p_build.add_argument("--index_bm25", default="data/bm25_index")
        p_build.add_argument("--embed_model", default="all-MiniLM-L6-v2")
        p_build.set_defaults(func=cmd_build_index)

        # generate_questions
        p_gq = sub.add_parser("generate_questions")
        p_gq.add_argument("--chunks_in", default="data/processed/chunks.jsonl")
        p_gq.add_argument("--out_questions", default="data/eval/questions.jsonl")
        p_gq.add_argument("--num_questions", type=int, default=100)
        p_gq.set_defaults(func=cmd_generate_questions)

        # run_eval
        p_eval = sub.add_parser("run_eval")
        p_eval.add_argument("--chunks_in", default="data/processed/chunks.jsonl")
        p_eval.add_argument("--index_dense", default="data/index.faiss")
        p_eval.add_argument("--index_bm25", default="data/bm25_index")
        p_eval.add_argument("--questions", default="data/eval/questions.jsonl")
        p_eval.add_argument("--out_prefix", default="results/report")
        p_eval.add_argument("--embed_model", default="all-MiniLM-L6-v2")
        p_eval.add_argument("--gen_model", default="google/flan-t5-base")
        p_eval.add_argument("--device", default="cpu")
        p_eval.add_argument("--top_n", type=int, default=10)
        p_eval.add_argument("--recall_k", type=int, default=10)
        p_eval.set_defaults(func=cmd_run_eval)

        # full pipeline
        p_full = sub.add_parser("full_pipeline")
        p_full.add_argument("--fixed", default="fixed_urls.json")
        p_full.add_argument("--generate_fixed", action="store_true")
        p_full.add_argument("--fixed_n", type=int, default=200)
        p_full.add_argument("--random_n", type=int, default=300)
        p_full.add_argument("--min_words", type=int, default=200)
        p_full.add_argument("--chunks_out", default="data/processed/chunks.jsonl")
        p_full.add_argument("--index_dense", default="data/index.faiss")
        p_full.add_argument("--index_bm25", default="data/bm25_index")
        p_full.add_argument("--out_questions", default="data/eval/questions.jsonl")
        p_full.add_argument("--out_prefix", default="results/report")
        p_full.add_argument("--embed_model", default="all-MiniLM-L6-v2")
        p_full.add_argument("--gen_model", default="google/flan-t5-base")
        p_full.add_argument("--device", default="cpu")
        p_full.add_argument("--top_n", type=int, default=10)
        p_full.add_argument("--recall_k", type=int, default=10)
        p_full.set_defaults(func=cmd_full_pipeline)

        args = parser.parse_args()
        if not hasattr(args, "func"):
            parser.print_help()
            return
        args.func(args)

    main()