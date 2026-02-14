#!/usr/bin/env python3
"""
Inter-rater and Intra-rater Reliability Analysis
Computes Cohen's Kappa and Fleiss' Kappa for LLM coding reliability
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.metrics import cohen_kappa_score
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Coding columns to analyze
CODING_COLS = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'T1', 'T2', 'Q1']

# Model configurations
MODELS = {
    'GPT-4o': {'runs': ['gpt4o_runs/run1.csv', 'gpt4o_runs/run2.csv', 'gpt4o_runs/run3.csv']},
    'GPT-5.2': {'runs': ['gpt52_runs/run1.csv', 'gpt52_runs/run2.csv', 'gpt52_runs/run3.csv']},
    'Gemini-3': {'runs': ['gemini3_runs/run1.csv', 'gemini3_runs/run2.csv', 'gemini3_runs/run3.csv']},
    'Sonnet-4.5': {'runs': ['sonnet45_runs/papers_coded_sonnet45.csv']},
    'Human': {'runs': ['human/papers_coded_verified_fixed.csv']}
}


def normalize_response(value):
    """
    Normalize categorical responses to handle variations
    - Convert to lowercase
    - Strip whitespace
    - Handle synonyms (None/N/A/empty)
    """
    if pd.isna(value):
        return 'n/a'

    # Convert to string and normalize
    value = str(value).strip().lower()

    # Handle empty or None-like values
    if value in ['', 'none', 'na', 'n/a', 'null']:
        return 'n/a'

    # Handle common variations
    value = value.replace('_', '-')  # underscores to hyphens
    value = ' '.join(value.split())  # normalize multiple spaces

    return value


def load_and_normalize_data(file_path):
    """Load CSV and normalize coding columns"""
    df = pd.read_csv(file_path)

    # Normalize only coding columns
    for col in CODING_COLS:
        if col in df.columns:
            df[col] = df[col].apply(normalize_response)

    return df


def majority_vote(values):
    """
    Get majority vote from list of values
    Returns most common value, or 'no-consensus' if all different
    """
    if len(values) == 0:
        return 'n/a'

    # Count occurrences
    counter = Counter(values)
    most_common = counter.most_common(1)[0]

    # If there's a clear majority (at least 2 out of 3 agree, or 2 out of 2)
    if most_common[1] >= 2 or len(values) == 2:
        return most_common[0]

    # All values different
    return 'no-consensus'


def create_consensus_version(model_name, runs):
    """
    Create consensus version from multiple runs using majority vote
    """
    dfs = [load_and_normalize_data(run) for run in runs]

    if len(dfs) == 1:
        # Only one run - just return it
        return dfs[0][['arxiv_id'] + CODING_COLS]

    # Multiple runs - create consensus
    consensus_df = dfs[0][['arxiv_id']].copy()

    for col in CODING_COLS:
        # Collect values from all runs for this column
        values_per_paper = []
        for _, row in consensus_df.iterrows():
            arxiv_id = row['arxiv_id']
            values = [df[df['arxiv_id'] == arxiv_id][col].values[0] for df in dfs]
            values_per_paper.append(majority_vote(values))

        consensus_df[col] = values_per_paper

    return consensus_df


def fleiss_kappa(ratings_matrix):
    """
    Calculate Fleiss' Kappa for multiple raters

    Args:
        ratings_matrix: numpy array of shape (n_items, n_categories)
                       where each cell contains the count of raters who assigned that category

    Returns:
        Fleiss' Kappa score
    """
    n_items, n_categories = ratings_matrix.shape
    n_raters = ratings_matrix.sum(axis=1)[0]  # Total raters per item (should be constant)

    # Proportion of all assignments in each category
    p_j = ratings_matrix.sum(axis=0) / (n_items * n_raters)

    # Calculate P_i (extent of agreement for each item)
    P_i = (ratings_matrix ** 2).sum(axis=1) - n_raters
    P_i = P_i / (n_raters * (n_raters - 1))

    # Mean of P_i
    P_bar = P_i.mean()

    # Expected agreement
    P_e = (p_j ** 2).sum()

    # Fleiss' Kappa
    if P_e == 1:
        return 1.0

    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa


def compute_fleiss_kappa_from_runs(runs, column=None):
    """
    Compute Fleiss' Kappa across multiple runs
    If column is None, computes across all coding columns merged
    """
    dfs = [load_and_normalize_data(run) for run in runs]
    n_runs = len(dfs)

    if n_runs < 2:
        return np.nan

    # Determine columns to analyze
    cols = [column] if column else CODING_COLS

    # Collect all values across runs and columns
    all_values = []
    for col in cols:
        for df in dfs:
            all_values.extend(df[col].values)

    # Get unique categories
    categories = sorted(set(all_values))
    n_categories = len(categories)

    # Number of items (papers * columns)
    n_papers = len(dfs[0])
    n_items = n_papers * len(cols)

    # Create ratings matrix
    ratings_matrix = np.zeros((n_items, n_categories))

    item_idx = 0
    for col in cols:
        for paper_idx in range(n_papers):
            # Count how many raters assigned each category for this item
            for df in dfs:
                value = df.iloc[paper_idx][col]
                cat_idx = categories.index(value)
                ratings_matrix[item_idx, cat_idx] += 1
            item_idx += 1

    return fleiss_kappa(ratings_matrix)


def compute_cohens_kappa_global(df1, df2):
    """
    Compute Cohen's Kappa between two dataframes across all coding columns merged
    """
    # Merge all coding columns into one long array per dataframe
    values1 = []
    values2 = []

    for col in CODING_COLS:
        values1.extend(df1[col].values)
        values2.extend(df2[col].values)

    return cohen_kappa_score(values1, values2)


def compute_cohens_kappa_per_column(df1, df2):
    """
    Compute Cohen's Kappa per column
    """
    kappas = {}
    for col in CODING_COLS:
        kappas[col] = cohen_kappa_score(df1[col].values, df2[col].values)
    return kappas


def main():
    print("=" * 80)
    print("INTER-RATER AND INTRA-RATER RELIABILITY ANALYSIS")
    print("=" * 80)
    print()

    # -------------------------------------------------------------------------
    # 1. Load and create consensus versions
    # -------------------------------------------------------------------------
    print("Loading data and creating consensus versions...")
    consensus_dfs = {}

    for model_name, config in MODELS.items():
        print(f"  - {model_name}: {len(config['runs'])} run(s)")
        consensus_dfs[model_name] = create_consensus_version(model_name, config['runs'])

    print()

    # -------------------------------------------------------------------------
    # 2. Compute INTRA-RATER reliability (diagonal)
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("INTRA-RATER RELIABILITY (Same model across runs)")
    print("=" * 80)
    print()

    intra_rater_scores = {}

    for model_name, config in MODELS.items():
        if len(config['runs']) >= 3:
            # Compute Fleiss' Kappa across all runs
            kappa = compute_fleiss_kappa_from_runs(config['runs'])
            intra_rater_scores[model_name] = kappa
            print(f"{model_name:15s}: Fleiss' Kappa = {kappa:.4f}")
        elif len(config['runs']) == 2:
            # Mark as N/A per user preference
            intra_rater_scores[model_name] = np.nan
            print(f"{model_name:15s}: N/A (only 2 runs)")
        else:
            intra_rater_scores[model_name] = np.nan
            print(f"{model_name:15s}: N/A (only 1 run)")

    print()

    # -------------------------------------------------------------------------
    # 3. Compute INTER-RATER reliability (off-diagonal)
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("INTER-RATER RELIABILITY (Between different models)")
    print("=" * 80)
    print()

    model_names = list(MODELS.keys())
    n_models = len(model_names)

    # Create agreement matrix
    agreement_matrix = np.zeros((n_models, n_models))

    # Fill diagonal with intra-rater scores
    for i, model_name in enumerate(model_names):
        agreement_matrix[i, i] = intra_rater_scores[model_name]

    # Fill off-diagonal with inter-rater Cohen's Kappa
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i < j:  # Only compute upper triangle
                kappa = compute_cohens_kappa_global(
                    consensus_dfs[model1],
                    consensus_dfs[model2]
                )
                agreement_matrix[i, j] = kappa
                agreement_matrix[j, i] = kappa  # Symmetric
                print(f"{model1:15s} vs {model2:15s}: Cohen's Kappa = {kappa:.4f}")

    print()

    # -------------------------------------------------------------------------
    # 4. Display Global Agreement Matrix
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("GLOBAL AGREEMENT MATRIX (All columns merged)")
    print("=" * 80)
    print()
    print("Diagonal: Intra-rater reliability (Fleiss' Kappa)")
    print("Off-diagonal: Inter-rater reliability (Cohen's Kappa)")
    print()

    # Create DataFrame for better display
    matrix_df = pd.DataFrame(
        agreement_matrix,
        index=model_names,
        columns=model_names
    )

    print(matrix_df.to_string(float_format=lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A"))
    print()

    # Save to CSV
    matrix_df.to_csv('global_agreement_matrix.csv')
    print("Saved to: global_agreement_matrix.csv")
    print()

    # -------------------------------------------------------------------------
    # 5. Per-Column Analysis
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("PER-COLUMN ANALYSIS")
    print("=" * 80)
    print()

    # Compute per-column agreement for each model pair
    per_column_results = []

    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i <= j:  # Include diagonal and upper triangle
                if i == j:
                    # Intra-rater per column
                    if len(MODELS[model1]['runs']) >= 3:
                        for col in CODING_COLS:
                            kappa = compute_fleiss_kappa_from_runs(MODELS[model1]['runs'], col)
                            per_column_results.append({
                                'Model_1': model1,
                                'Model_2': model1,
                                'Type': 'Intra-rater',
                                'Column': col,
                                'Kappa': kappa
                            })
                else:
                    # Inter-rater per column
                    kappas = compute_cohens_kappa_per_column(
                        consensus_dfs[model1],
                        consensus_dfs[model2]
                    )
                    for col, kappa in kappas.items():
                        per_column_results.append({
                            'Model_1': model1,
                            'Model_2': model2,
                            'Type': 'Inter-rater',
                            'Column': col,
                            'Kappa': kappa
                        })

    per_column_df = pd.DataFrame(per_column_results)
    per_column_df.to_csv('per_column_agreement.csv', index=False)
    print("Per-column agreement saved to: per_column_agreement.csv")
    print()

    # Display summary statistics per column
    print("Average Kappa per Column (across all model pairs):")
    print("-" * 40)
    for col in CODING_COLS:
        col_data = per_column_df[per_column_df['Column'] == col]['Kappa']
        avg_kappa = col_data.mean()
        print(f"{col:5s}: {avg_kappa:.4f} (std: {col_data.std():.4f})")

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
