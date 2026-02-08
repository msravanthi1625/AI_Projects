import os
import json
import pandas as pd
from typing import List, Dict, Tuple
import argparse
# Import from hybrid_rag_eval
from hybrid_rag_eval import (
    DenseIndexer, BM25Indexer, Retriever, Generator, Evaluator, QAItem
)

def run_ablation_study(
    chunks_path: str,
    index_dense: str,
    index_bm25: str,
    questions_path: str,
    out_dir: str = "results/ablation",
    embed_model: str = "all-MiniLM-L6-v2",
    gen_model: str = "google/flan-t5-base",
    device: str = "cpu",
    top_n: int = 10,
    recall_k: int = 10,
    k_dense_values: List[int] = None,
    k_sparse_values: List[int] = None,
    rrf_k_values: List[int] = None
):
    """
    Run ablation study across different k_dense, k_sparse, and rrf_k combinations.
    Exports results to CSV and summary JSON.
    """
    if top_n is None:
        top_n = [3, 5, 10]
    if k_dense_values is None:
        k_dense_values = [50, 100, 200]
    if k_sparse_values is None:
        k_sparse_values = [50, 100, 200]
    if rrf_k_values is None:
        rrf_k_values = [30, 60, 120]

    os.makedirs(out_dir, exist_ok=True)

    # Load indexes and models once
    print("Loading indexes and models...")
    dense = DenseIndexer(model_name=embed_model)
    dense.load_index(index_dense)
    bm25 = BM25Indexer()
    bm25.load(index_bm25)
    retriever = Retriever(dense, bm25, chunks_path)
    generator = Generator(model_name=gen_model, device=device)

    # Load QA items
    qa_items = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qa_items.append(QAItem(**obj))
    print(f"Loaded {len(qa_items)} QA items")

    evaluator = Evaluator()

    # Run ablations
    ablation_results = []
    total_combos = len(k_dense_values) * len(k_sparse_values) * len(rrf_k_values)
    combo_count = 0

    for t_n in top_n:
        for k_d in k_dense_values:
            for k_s in k_sparse_values:
                for rrf_k in rrf_k_values:
                    combo_count += 1
                    print(f"\n[{combo_count}/{total_combos}] Running: top_n={t_n}, k_dense={k_d}, k_sparse={k_s}, rrf_k={rrf_k}")

                    # Run evaluation for this combo
                    df_eval, agg_metrics = evaluator.evaluate_all(
                        qa_items,
                        retriever,
                        generator,
                        top_n=t_n,
                        recall_k=recall_k,
                        k_dense=k_d,
                        k_sparse=k_s,
                        rrf_k=rrf_k
                    )

                    # Create summary row for ablation table
                    summary_row = {
                        "top_n": t_n,
                        "k_dense": k_d,
                        "k_sparse": k_s,
                        "rrf_k": rrf_k,
                        "config": f"dense_{k_d}_sparse_{k_s}_rrf_{rrf_k}",
                        **agg_metrics  # MRR_URL, Exact_Match, Recall@K, Avg_Time
                    }
                    ablation_results.append(summary_row)

                    # Optionally save per-combo detailed results
                    combo_prefix = os.path.join(
                        out_dir,
                        f"top_n_{t_n}_dense_{k_d}_sparse_{k_s}_rrf_{rrf_k}"
                    )
                    df_eval.to_csv(combo_prefix + ".csv", index=False)
                    print(f"  → Saved {combo_prefix}.csv")

    # Export ablation summary table (all combos)
    df_ablation = pd.DataFrame(ablation_results)
    ablation_csv = os.path.join(out_dir, "ablation_summary.csv")
    df_ablation.to_csv(ablation_csv, index=False)
    print(f"\nExported ablation summary to {ablation_csv}")

    # Export as JSON for downstream analysis
    ablation_json = os.path.join(out_dir, "ablation_summary.json")
    with open(ablation_json, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"Exported ablation summary to {ablation_json}")

    # Print summary table to console
    print("\n=== ABLATION STUDY SUMMARY ===")
    print(df_ablation.to_string(index=False))

    return df_ablation


def run_component_ablation(
    chunks_path: str,
    index_dense: str,
    index_bm25: str,
    questions_path: str,
    out_dir: str = "results/ablation",
    embed_model: str = "all-MiniLM-L6-v2",
    gen_model: str = "google/flan-t5-base",
    device: str = "cpu",
    top_n: int = 10,
    recall_k: int = 10
):
    """
    Compare dense-only, sparse-only, and hybrid (RRF) retrieval.
    """
    os.makedirs(out_dir, exist_ok=True)

    print("Loading indexes and models...")
    dense = DenseIndexer(model_name=embed_model)
    dense.load_index(index_dense)
    bm25 = BM25Indexer()
    bm25.load(index_bm25)
    retriever = Retriever(dense, bm25, chunks_path)
    generator = Generator(model_name=gen_model, device=device)

    qa_items = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qa_items.append(QAItem(**obj))
    print(f"Loaded {len(qa_items)} QA items")

    evaluator = Evaluator()

    # Define component configurations
    configs = [
        {"name": "dense_only", "k_dense": 100, "k_sparse": 0, "rrf_k": 60},
        {"name": "sparse_only", "k_dense": 0, "k_sparse": 100, "rrf_k": 60},
        {"name": "hybrid_rrf", "k_dense": 100, "k_sparse": 100, "rrf_k": 60}
    ]

    component_results = []

    for cfg in configs:
        name = cfg["name"]
        k_d = cfg["k_dense"]
        k_s = cfg["k_sparse"]
        rrf_k = cfg["rrf_k"]

        print(f"\nRunning component test: {name} (k_dense={k_d}, k_sparse={k_s}, rrf_k={rrf_k})")

        # Custom retrieve logic for dense-only / sparse-only
        if name == "dense_only":
            # use dense results directly (skip RRF)
            rows = []
            mrrs, ems, recalls, times = [], [], [], []
            for qa in qa_items:
                import time as time_module
                start = time_module.time()
                dense_res = dense.search(qa.question, k=k_d)
                retrieved_urls = [cid for cid, _ in dense_res]
                # map chunk ids to urls
                chunk_map = {}
                with open(chunks_path, "r", encoding="utf-8") as f:
                    for line in f:
                        obj = json.loads(line)
                        chunk_map[obj["id"]] = obj

                context_texts = [chunk_map.get(cid, {}).get("text", "") for cid, _ in dense_res]
                answer, _ = generator.generate(qa.question, context_texts)
                elapsed = time_module.time() - start

                mrr = evaluator.compute_mrr_url(qa.answer_url, retrieved_urls)
                em = evaluator.exact_match(answer, qa.answer)
                recall = evaluator.recall_at_k(qa.answer_url, retrieved_urls, k=recall_k)

                rows.append({
                    "qid": qa.qid,
                    "method": name,
                    "predicted_answer": answer,
                    "mrr_url": mrr,
                    "exact_match": em,
                    "recall_at_k": recall,
                    "time": elapsed
                })
                mrrs.append(mrr)
                ems.append(em)
                recalls.append(recall)
                times.append(elapsed)

            df_res = pd.DataFrame(rows)
            agg = {
                "MRR_URL": float(pd.Series(mrrs).mean()),
                "Exact_Match": float(pd.Series(ems).mean()),
                f"Recall@{recall_k}": float(pd.Series(recalls).mean()),
                "Avg_Time": float(pd.Series(times).mean())
            }

        elif name == "sparse_only":
            # use sparse results directly
            rows = []
            mrrs, ems, recalls, times = [], [], [], []
            for qa in qa_items:
                import time as time_module
                start = time_module.time()
                sparse_res = bm25.search(qa.question, k=k_s)
                retrieved_urls = [cid for cid, _ in sparse_res]
                chunk_map = {}
                with open(chunks_path, "r", encoding="utf-8") as f:
                    for line in f:
                        obj = json.loads(line)
                        chunk_map[obj["id"]] = obj

                context_texts = [chunk_map.get(cid, {}).get("text", "") for cid, _ in sparse_res]
                answer, _ = generator.generate(qa.question, context_texts)
                elapsed = time_module.time() - start

                mrr = evaluator.compute_mrr_url(qa.answer_url, retrieved_urls)
                em = evaluator.exact_match(answer, qa.answer)
                recall = evaluator.recall_at_k(qa.answer_url, retrieved_urls, k=recall_k)

                rows.append({
                    "qid": qa.qid,
                    "method": name,
                    "predicted_answer": answer,
                    "mrr_url": mrr,
                    "exact_match": em,
                    "recall_at_k": recall,
                    "time": elapsed
                })
                mrrs.append(mrr)
                ems.append(em)
                recalls.append(recall)
                times.append(elapsed)

            df_res = pd.DataFrame(rows)
            agg = {
                "MRR_URL": float(pd.Series(mrrs).mean()),
                "Exact_Match": float(pd.Series(ems).mean()),
                f"Recall@{recall_k}": float(pd.Series(recalls).mean()),
                "Avg_Time": float(pd.Series(times).mean())
            }

        else:  # hybrid_rrf
            df_res, agg = evaluator.evaluate_all(qa_items, retriever, generator, top_n=top_n, recall_k=recall_k)

        # Save detailed results for this component
        csv_path = os.path.join(out_dir, f"component_{name}.csv")
        df_res.to_csv(csv_path, index=False)
        print(f"  → Saved {csv_path}")

        # Add to summary
        summary = {"method": name, **agg}
        component_results.append(summary)

    # Export component summary
    df_components = pd.DataFrame(component_results)
    component_csv = os.path.join(out_dir, "component_comparison.csv")
    df_components.to_csv(component_csv, index=False)
    print(f"\nExported component comparison to {component_csv}")

    print("\n=== COMPONENT ABLATION SUMMARY ===")
    print(df_components.to_string(index=False))

    return df_components


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study for hybrid RAG system")
    subparsers = parser.add_subparsers(dest="study_type")

    # Hyperparameter ablation
    p_hyper = subparsers.add_parser("hyperparameter", help="Ablation over k_dense, k_sparse, rrf_k")
    p_hyper.add_argument("--chunks_in", default="data/processed/chunks.jsonl")
    p_hyper.add_argument("--index_dense", default="data/index.faiss")
    p_hyper.add_argument("--index_bm25", default="data/bm25_index")
    p_hyper.add_argument("--questions", default="data/eval/questions.jsonl")
    p_hyper.add_argument("--out_dir", default="results/ablation")
    p_hyper.add_argument("--embed_model", default="all-MiniLM-L6-v2")
    p_hyper.add_argument("--gen_model", default="google/flan-t5-base")
    p_hyper.add_argument("--device", default="cpu")
    p_hyper.add_argument("--top_n", nargs="+", type=int, default=[3, 5, 10])
    p_hyper.add_argument("--recall_k", type=int, default=10)
    p_hyper.add_argument("--k_dense", nargs="+", type=int, default=[50, 100, 200])
    p_hyper.add_argument("--k_sparse", nargs="+", type=int, default=[50, 100, 200])
    p_hyper.add_argument("--rrf_k", nargs="+", type=int, default=[30, 60, 120])

    # Component ablation
    p_comp = subparsers.add_parser("component", help="Compare dense-only, sparse-only, hybrid")
    p_comp.add_argument("--chunks_in", default="data/processed/chunks.jsonl")
    p_comp.add_argument("--index_dense", default="data/index.faiss")
    p_comp.add_argument("--index_bm25", default="data/bm25_index")
    p_comp.add_argument("--questions", default="data/eval/questions.jsonl")
    p_comp.add_argument("--out_dir", default="results/ablation")
    p_comp.add_argument("--embed_model", default="all-MiniLM-L6-v2")
    p_comp.add_argument("--gen_model", default="google/flan-t5-base")
    p_comp.add_argument("--device", default="cpu")
    p_comp.add_argument("--top_n", type=int, default=10)
    p_comp.add_argument("--recall_k", type=int, default=10)

    args = parser.parse_args()

    if args.study_type == "hyperparameter":
        run_ablation_study(
            chunks_path=args.chunks_in,
            index_dense=args.index_dense,
            index_bm25=args.index_bm25,
            questions_path=args.questions,
            out_dir=args.out_dir,
            embed_model=args.embed_model,
            gen_model=args.gen_model,
            device=args.device,
            top_n=args.top_n,
            recall_k=args.recall_k,
            k_dense_values=args.k_dense,
            k_sparse_values=args.k_sparse,
            rrf_k_values=args.rrf_k
        )
    elif args.study_type == "component":
        run_component_ablation(
            chunks_path=args.chunks_in,
            index_dense=args.index_dense,
            index_bm25=args.index_bm25,
            questions_path=args.questions,
            out_dir=args.out_dir,
            embed_model=args.embed_model,
            gen_model=args.gen_model,
            device=args.device,
            top_n=args.top_n,
            recall_k=args.recall_k
        )
    else:
        parser.print_help()