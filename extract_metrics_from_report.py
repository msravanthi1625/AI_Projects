import pandas as pd
import re
import os
import glob

def parse_params_from_filename(filename):
    pattern = r"top_n_(\d+)_dense_(\d+)_sparse_(\d+)_rrf_(\d+)"
    match = re.search(pattern, filename)
    if not match:
        return None

    return {
        "top_n": int(match.group(1)),
        "dense_k": int(match.group(2)),
        "sparse_k": int(match.group(3)),
        "rrf_k": int(match.group(4)),
    }

def compute_metrics(csv_path):
    df = pd.read_csv(csv_path)
    return {
        "MRR_URL": df["mrr_url"].mean(),
        "Exact_Match": df["exact_match"].mean(),
        "Recall@10": df["recall_at_k"].mean(),
        "Avg_Time": df["time"].mean(),
    }

# -------- main --------
REPORTS_DIR = "results/ablation"
OUTPUT_CSV = "results/ablation/summary.csv"

rows = []

for csv_path in glob.glob(os.path.join(REPORTS_DIR, "*.csv")):
    filename = os.path.basename(csv_path)

    params = parse_params_from_filename(filename)
    if params is None:
        continue

    metrics = compute_metrics(csv_path)
    rows.append({**params, **metrics})

if not rows:
    raise RuntimeError("No matching report files found.")

summary_df = pd.DataFrame(rows)

# 🔑 Sort by hyperparameters (in the requested order)
summary_df = summary_df.sort_values(
    by=["top_n", "dense_k", "sparse_k", "rrf_k"],
    ascending=[True, True, True, True]
)

summary_df.to_csv(OUTPUT_CSV, index=False)

print(f"Wrote {len(summary_df)} rows to {OUTPUT_CSV}")
