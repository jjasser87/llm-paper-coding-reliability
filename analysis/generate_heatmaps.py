#!/usr/bin/env python3
"""Generate reliability heatmaps from analysis results"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
global_matrix = pd.read_csv('global_agreement_matrix.csv', index_col=0)
per_column = pd.read_csv('per_column_agreement.csv')

# Model names for display
model_names = ['GPT-4o', 'GPT-5.2', 'Gemini-3', 'Sonnet-4.5', 'Human']

# =============================================================================
# Figure 1: Global Agreement Heatmap
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Create mask for NaN (diagonal for single-run models)
mask = np.zeros_like(global_matrix.values, dtype=bool)

# Create annotation labels
annot = global_matrix.values.copy()
annot_labels = np.empty_like(annot, dtype=object)
for i in range(len(model_names)):
    for j in range(len(model_names)):
        if np.isnan(annot[i, j]):
            annot_labels[i, j] = ''
        else:
            annot_labels[i, j] = f'{annot[i, j]:.3f}'

# Plot heatmap without annotations first
heatmap = sns.heatmap(global_matrix,
                      annot=False,
                      cmap='RdYlGn',
                      vmin=0, vmax=1,
                      square=True,
                      linewidths=0.5,
                      cbar_kws={'label': ''},
                      ax=ax)

# Manually add text annotations to ensure they all appear
for i in range(len(model_names)):
    for j in range(len(model_names)):
        text = annot_labels[i, j]
        if text:
            # Determine text color based on background value
            val = annot[i, j]
            text_color = 'white' if val > 0.5 else 'black'
            ax.text(j + 0.5, i + 0.5, text,
                   ha='center', va='center',
                   fontsize=11, fontweight='bold',
                   color=text_color)

ax.set_title("Inter-Rater and Intra-Rater Reliability Matrix\n(Cohen's Kappa & Fleiss' Kappa)",
             fontsize=14, fontweight='bold')
ax.set_xlabel('Rater', fontsize=12)
ax.set_ylabel('Rater', fontsize=12)

# Add footnote
fig.text(0.5, 0.02,
         "Diagonal: Intra-rater reliability (Fleiss' Kappa) | Off-diagonal: Inter-rater reliability (Cohen's Kappa)\n"
         "Color scale: Red (poor) → Yellow (moderate) → Green (excellent)",
         ha='center', fontsize=10, style='italic', color='#555555')

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.savefig('reliability_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

print("Generated: reliability_heatmap.png")

# =============================================================================
# Figure 2: Per-Column Heatmaps
# =============================================================================
coding_cols = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'T1', 'T2', 'Q1']

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for idx, col in enumerate(coding_cols):
    ax = axes[idx]

    # Create matrix for this column
    col_matrix = np.full((5, 5), np.nan)

    # Fill from per_column data
    for _, row in per_column[per_column['Column'] == col].iterrows():
        m1 = model_names.index(row['Model_1'])
        m2 = model_names.index(row['Model_2'])
        col_matrix[m1, m2] = row['Kappa']
        if m1 != m2:
            col_matrix[m2, m1] = row['Kappa']

    # Create annotation labels
    annot_labels = np.empty_like(col_matrix, dtype=object)
    for i in range(5):
        for j in range(5):
            if np.isnan(col_matrix[i, j]):
                annot_labels[i, j] = ''
            else:
                annot_labels[i, j] = f'{col_matrix[i, j]:.2f}'

    # Plot without annotations first
    sns.heatmap(col_matrix,
                annot=False,
                cmap='RdYlGn',
                vmin=0, vmax=1,
                square=True,
                linewidths=0.5,
                cbar=False,
                xticklabels=model_names,
                yticklabels=model_names,
                ax=ax)

    # Manually add text annotations
    for i in range(5):
        for j in range(5):
            text = annot_labels[i, j]
            if text:
                # Determine text color based on background value
                val = col_matrix[i, j]
                text_color = 'white' if val > 0.5 else 'black'
                ax.text(j + 0.5, i + 0.5, text,
                       ha='center', va='center',
                       fontsize=9, fontweight='bold',
                       color=text_color)

    ax.set_title(col, fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

fig.suptitle("Per-Column Agreement (Cohen's Kappa)", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('per_column_heatmaps.png', dpi=150, bbox_inches='tight')
plt.close()

print("Generated: per_column_heatmaps.png")
print("\nDone!")
