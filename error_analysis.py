"""
Error Analysis Script for RAG System Evaluation

Categorizes failures (retrieval, generation, context issues) by question type
with comprehensive visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
from pathlib import Path
import re

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Load the evaluation questions json file
qid_to_category = {}

with open("data/eval/questions.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        qid_to_category[obj["qid"]] = obj["category"]

def extract_question_type(qid):
    """Extract question type from qid using the mapping."""
    return qid_to_category.get(qid, "Unknown")

def classify_failure(row):
    """
    Classify the type of failure for a given question.
    
    Returns: 'Success', 'Retrieval Issue', 'Generation Issue', 'Context Issue', or 'Complete Miss'
    """
    exact_match = row['exact_match']
    
    # Success case
    if exact_match == 1:
        return 'Success'
    
    # Extract the URLs from the retrieved_urls string
    try:
        retrieved_urls_str = row['retrieved_urls']
        if isinstance(retrieved_urls_str, str):
            # Parse the list-like string
            retrieved_urls = eval(retrieved_urls_str)
        else:
            retrieved_urls = []
    except:
        retrieved_urls = []
    
    gold_url = str(row['gold_url']).strip()
    
    # Check if gold URL was retrieved
    gold_url_retrieved = any(gold_url in url for url in retrieved_urls)
    
    if not gold_url_retrieved:
        # URL not retrieved = Retrieval issue
        return 'Retrieval Issue'
    else:
        # URL retrieved but answer still wrong = Generation/Context issue
        # This means the retriever found the right context but generator failed
        if row['mrr_url'] == 0.0:
            # MRR is 0 but somehow marked as retrieved - unclear retrieval
            return 'Retrieval Issue'
        else:
            # Good retrieval but generation failed
            return 'Generation Issue'


def load_and_analyze_report(report_path):
    """Load the report and perform error analysis."""
    
    # Load the CSV file
    df = pd.read_csv(report_path)
    
    print(f"Total questions: {len(df)}")
    
    # Add new columns for analysis
    df['Question_Type'] = df['qid'].apply(extract_question_type)
    df['Failure_Type'] = df.apply(classify_failure, axis=1)
    
    # Create failure status (Success vs any failure)
    df['Status'] = df['Failure_Type'].apply(lambda x: 'Success' if x == 'Success' else 'Failed')
    
    return df


def create_failure_summary(df):
    """Create summary statistics for failures."""
    print("\n" + "="*80)
    print("FAILURE ANALYSIS SUMMARY")
    print("="*80)
    
    # Overall statistics
    total = len(df)
    success = (df['Failure_Type'] == 'Success').sum()
    retrieval_issues = (df['Failure_Type'] == 'Retrieval Issue').sum()
    generation_issues = (df['Failure_Type'] == 'Generation Issue').sum()
    
    print(f"\nOverall Performance:")
    print(f"  Total: {total}")
    print(f"  Success: {success} ({success/total*100:.1f}%)")
    print(f"  Retrieval Issues: {retrieval_issues} ({retrieval_issues/total*100:.1f}%)")
    print(f"  Generation Issues: {generation_issues} ({generation_issues/total*100:.1f}%)")
    
    # By question type
    print(f"\nFailure Analysis by Question Type:")
    print("-" * 80)
    
    qa_types = df['Question_Type'].unique()
    results = []
    
    for qtype in sorted(qa_types):
        type_df = df[df['Question_Type'] == qtype]
        total_type = len(type_df)
        success_type = (type_df['Failure_Type'] == 'Success').sum()
        retrieval_type = (type_df['Failure_Type'] == 'Retrieval Issue').sum()
        generation_type = (type_df['Failure_Type'] == 'Generation Issue').sum()
        
        results.append({
            'Question_Type': qtype,
            'Total': total_type,
            'Success': success_type,
            'Success_Rate': f"{success_type/total_type*100:.1f}%",
            'Retrieval_Issues': retrieval_type,
            'Generation_Issues': generation_type
        })
        
        print(f"  {qtype:20} | Total: {total_type:3} | Success: {success_type:3} ({success_type/total_type*100:5.1f}%) | Retrieval: {retrieval_type:2} | Generation: {generation_type:2}")
    
    return pd.DataFrame(results)


def create_visualizations(df, output_dir='results/error_analysis'):
    """Create comprehensive visualizations."""
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Failure Type Distribution (Pie Chart)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    failure_counts = df['Failure_Type'].value_counts()
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    
    axes[0].pie(failure_counts.values, labels=failure_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[0].set_title('Overall Failure Type Distribution', fontsize=14, fontweight='bold')
    
    # Status distribution
    status_counts = df['Status'].value_counts()
    axes[1].bar(status_counts.index, status_counts.values, color=['#2ecc71', '#e74c3c'])
    axes[1].set_title('Success vs Failed', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Count')
    for i, v in enumerate(status_counts.values):
        axes[1].text(i, v + 1, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_failure_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/01_failure_distribution.png")
    plt.close()
    
    # 2. Failure Type by Question Type (Stacked Bar Chart)
    fig, ax = plt.subplots(figsize=(14, 6))
    
    cross_tab = pd.crosstab(df['Question_Type'], df['Failure_Type'])
    cross_tab_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100
    
    cross_tab_pct.plot(kind='bar', stacked=True, ax=ax, 
                       color=['#2ecc71', '#f39c12', '#e74c3c'])
    ax.set_title('Failure Type Distribution by Question Type (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Question Type', fontsize=12)
    ax.set_ylabel('Percentage', fontsize=12)
    ax.legend(title='Failure Type', loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_failure_by_question_type.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/02_failure_by_question_type.png")
    plt.close()
    
    # 3. Success Rate by Question Type (Bar Chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    success_by_type = df.groupby('Question_Type').apply(
        lambda x: (x['Failure_Type'] == 'Success').sum() / len(x) * 100
    ).sort_values(ascending=False)
    
    bars = ax.barh(success_by_type.index, success_by_type.values, 
                   color=['#2ecc71' if x > 50 else '#f39c12' if x > 30 else '#e74c3c' 
                          for x in success_by_type.values])
    ax.set_xlabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate by Question Type', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, success_by_type.values)):
        ax.text(value + 2, i, f'{value:.1f}%', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_success_rate_by_question_type.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/03_success_rate_by_question_type.png")
    plt.close()
    
    # 4. Response Time Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    failure_types = df['Failure_Type'].unique()
    data_by_failure = [df[df['Failure_Type'] == ft]['time'].values for ft in sorted(failure_types)]
    
    axes[0].boxplot(data_by_failure, labels=sorted(failure_types))
    axes[0].set_title('Response Time by Failure Type', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Time (seconds)', fontsize=12)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Scatter plot
    scatter_colors = {'Success': '#2ecc71', 'Retrieval Issue': '#e74c3c', 'Generation Issue': '#f39c12'}
    for failure_type in df['Failure_Type'].unique():
        mask = df['Failure_Type'] == failure_type
        axes[1].scatter(df[mask].index, df[mask]['time'], 
                       label=failure_type, alpha=0.6, s=50,
                       color=scatter_colors.get(failure_type, '#95a5a6'))
    
    axes[1].set_title('Response Time Over Questions', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Question ID', fontsize=12)
    axes[1].set_ylabel('Time (seconds)', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_response_time_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/04_response_time_analysis.png")
    plt.close()
    
    # 5. Question Type Distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    qtype_counts = df['Question_Type'].value_counts().sort_values(ascending=True)
    ax.barh(qtype_counts.index, qtype_counts.values, color='#3498db')
    ax.set_xlabel('Count', fontsize=12)
    ax.set_title('Question Type Distribution', fontsize=14, fontweight='bold')
    
    for i, v in enumerate(qtype_counts.values):
        ax.text(v + 0.5, i, str(v), va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_question_type_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/05_question_type_distribution.png")
    plt.close()
    
    # 6. Heatmap: Question Type vs Failure Type
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cross_tab = pd.crosstab(df['Question_Type'], df['Failure_Type'])
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title('Failure Count Heatmap: Question Type vs Failure Type', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_heatmap_question_vs_failure.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/06_heatmap_question_vs_failure.png")
    plt.close()
    
    print("\n✓ All visualizations saved to:", output_dir)


def export_detailed_errors(df, output_dir='results/error_analysis'):
    """Export detailed error cases for manual review."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get failed cases
    failed_df = df[df['Status'] == 'Failed'].copy()
    
    # Group by failure type
    for failure_type in failed_df['Failure_Type'].unique():
        subset = failed_df[failed_df['Failure_Type'] == failure_type]
        
        # Create a readable format
        export_data = []
        for _, row in subset.iterrows():
            export_data.append({
                'QID': row['qid'],
                'Question_Type': row['Question_Type'],
                'Question': row['question'][:100],  # First 100 chars
                'Gold_Answer': row['gold_answer'][:100],
                'Predicted_Answer': row['predicted_answer'][:100],
                'MRR': row['mrr_url'],
                'Recall@K': row['recall_at_k'],
                'Response_Time': f"{row['time']:.2f}s"
            })
        
        export_df = pd.DataFrame(export_data)
        filename = f'{output_dir}/error_details_{failure_type.lower().replace(" ", "_")}.csv'
        export_df.to_csv(filename, index=False)
        print(f"✓ Saved: {filename}")


def main():
    """Main execution function."""
    
    report_path = 'results/report.csv'
    
    # Check if file exists
    if not Path(report_path).exists():
        print(f"Error: {report_path} not found!")
        return
    
    print("Loading and analyzing report...")
    df = load_and_analyze_report(report_path)
    
    # Create summary
    summary_df = create_failure_summary(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(df)
    
    # Export detailed errors
    print("\nExporting detailed error cases...")
    export_detailed_errors(df)
    
    # Save summary report
    summary_path = 'results/error_analysis/error_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved: {summary_path}")
    
    print("\n" + "="*80)
    print("Error analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
