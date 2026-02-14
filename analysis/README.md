# Reliability Analysis Results

**71 papers | 9 coding categories (G1-G6, T1-T2, Q1) | 5 raters**

---

## Methodology (for paper)

### 1. Corpus

- **N = 71** academic papers in adversarial machine learning (AML).
- Papers were selected from a larger set using explicit criteria (e.g., presence in multiple artifact ecosystems, regulatory relevance); each has `arxiv_id`, `paper_title`, and metadata. The same ordered list of 71 papers was used for every rater.
- For LLM raters, full-text input was obtained by extracting text from PDFs (PyMuPDF) stored in a single `papers_for_coding` directory; paper identity was matched by title and arXiv ID.

### 2. Coding schema (9 variables)

Each paper was coded on exactly **9 categorical variables**. Allowed values were fixed and shared across LLM runs; the schema was derived from a human-verified coding pass and enforced in prompts and validation.

| Column | Description | Allowed values (summary) |
|--------|-------------|--------------------------|
| **G1** | Paper type | Attack, Defense, Evaluation, Both, Attack/Defense, Attack/Evaluation, Defense/Evaluation, Attack/Defense/Evaluation |
| **G2** | Threat category | Evasion, Poisoning, Privacy, Defense, N/A |
| **G3** | Application domain | Vision, NLP, LLM, Audio, Malware, Tabular, Cross, Cross-domain |
| **G4** | Publication venue | ML, Security, Journal, arXiv-only |
| **G5** | Code available | Yes, No |
| **G6** | Code release timing | At-pub, Post-pub, Never |
| **T1** | Model access (attacks) | White, Black, Gray, White/Black, N/A |
| **T2** | Gradient required (attacks) | Yes, No, N/A |
| **Q1** | Real-world evaluation | Yes, No, Partial, N/A |

### 3. Raters and runs

| Rater | API / source | Runs used | Runs per model | Consensus |
|-------|----------------|-----------|----------------|-----------|
| **GPT-4o** | OpenAI | run1, run2, run3 | **3** | Majority vote |
| **GPT-5.2** | OpenAI | run1, run2, run3 | **3** | Majority vote |
| **Gemini-3** | Google (Gemini 3.0 Pro) | run1, run2, run3 | **3** | Majority vote |
| **Claude Sonnet 4.5** | Anthropic (claude-sonnet-4-5-20250929) | run2 | **1** | N/A |
| **Human** | Expert coder | single verified file | **1** | N/A |

- **LLM runs:** Each run was an independent execution of the same coding script: same 71-paper list, same system/user prompt (with strict allowed values), same PDF text extraction pipeline. Temperature was set to **0** for all LLM runs to reduce within-model variability. Runs were executed sequentially with rate limiting (e.g., 1.5 s between papers for GPT-4o). Outputs were written to `run1.csv`, `run2.csv`, `run3.csv` per model. Sonnet-4.5 contributed a single run (run2) to the analysis.
- **Human:** A single expert coding of the 71 papers was performed, then corrected using a structured corrections file (`coding_corrections.csv`); the final **human reference** is `papers_coded_verified_fixed.csv`.

### 4. Preprocessing and consensus

- **Normalization (all raters):** Before any agreement computation, each coding value was normalized: lowercase, whitespace stripped, underscores replaced with hyphens; and `None`, `N/A`, `null`, or empty string were mapped to a single category `n/a`.
- **Consensus for multi-run raters:** For GPT-4o, GPT-5.2, and Gemini-3, a single consensus label per (paper, variable) was formed by **majority vote** across that rater’s runs (e.g., if ≥2 of 3 runs agreed, that value was used). Ties or no clear majority were handled as in the analysis script (e.g., “no-consensus” where applicable). Sonnet-4.5 and Human each contributed a single run; no consensus step was applied. All raters’ codings were then aligned by `arxiv_id` and the 9 columns for agreement analysis.

### 5. Reliability statistics

- **Intra-rater (same model, multiple runs):** For raters with **≥3 runs** (GPT-4o, GPT-5.2, Gemini-3), **Fleiss’ κ** was computed over the 3 runs, with items defined as (paper × variable) and raters as the 3 runs. Sonnet-4.5 (1 run) and Human (1 run) have no intra-rater κ in the matrix.
- **Inter-rater (between raters):** For every pair of raters, **Cohen’s κ** was computed on the consensus (or single) codings.
- **Global agreement matrix:** The **diagonal** is Fleiss’ κ (intra-rater) for that model; the **off-diagonal** is Cohen’s κ (inter-rater) for that pair. Global metrics were computed by **merging all 9 coding columns** into one long sequence of labels per rater (639 = 71×9 items per rater), then computing κ on those concatenated vectors.
- **Per-category agreement:** Cohen’s κ was also computed **per coding variable** (G1–G6, T1, T2, Q1) for each pair of raters; results are in `per_column_agreement.csv` and summarized in “Agreement by coding category” below.

### 6. Reproducibility

- **Script:** `reliability_analysis.py` in this directory loads the run CSVs and human file listed above, applies normalization and majority-vote consensus, then computes Fleiss’ and Cohen’s κ and writes `global_agreement_matrix.csv` and `per_column_agreement.csv`. Heatmaps are produced from these outputs.
- **Model run scripts:** Each LLM has a dedicated folder (`gpt4o_runs/`, `gpt52_runs/`, `gemini3_runs/`, `sonnet45_runs/`) with a `code_papers_*.py` script and README describing how to reproduce run1/run2/run3 (Sonnet-4.5 analysis uses a single run: run2).

---

### Normalization (summary)
All responses normalized before comparison:
- Lowercase conversion
- Whitespace stripped
- Synonyms unified: `None`, `N/A`, `null`, empty → `n/a`

### Consensus (Majority Vote)
For models with multiple runs: if ≥2 runs agree, use that value. All coding columns merged for global metrics.

### Statistical Measures
| Metric | Use Case | Applied To |
|--------|----------|------------|
| **Cohen's Kappa** | Pairwise inter-rater agreement | Off-diagonal (between different models) |
| **Fleiss' Kappa** | Multi-rater agreement | Diagonal (same model across 3 runs) |

---

## Global Agreement Matrix

|            | GPT-4o | GPT-5.2 | Gemini-3 | Sonnet-4.5 | Human |
|------------|--------|---------|----------|------------|-------|
| **GPT-4o**     | 0.975  | 0.803   | 0.787    | 0.343      | 0.654 |
| **GPT-5.2**    | 0.803  | 0.975   | 0.846    | 0.346      | 0.675 |
| **Gemini-3**   | 0.787  | 0.846   | 1.000    | 0.341      | 0.707 |
| **Sonnet-4.5** | 0.343  | 0.346   | 0.341    | N/A        | 0.270 |
| **Human**      | 0.654  | 0.675   | 0.707    | 0.270      | N/A   |

- **Diagonal**: Intra-rater (Fleiss' κ)
- **Off-diagonal**: Inter-rater (Cohen's κ)

---

## Key Results

### Intra-rater Reliability
| Model | Runs | Fleiss' κ |
|-------|------|-----------|
| Gemini-3 | 3 | 1.000 |
| GPT-5.2 | 3 | 0.975 |
| GPT-4o | 3 | 0.975 |

### Best Inter-rater Pairs
| Pair | Cohen's κ |
|------|-----------|
| GPT-5.2 ↔ Gemini-3 | 0.846 |
| GPT-4o ↔ GPT-5.2 | 0.803 |
| GPT-4o ↔ Gemini-3 | 0.787 |
| Gemini-3 ↔ Human | 0.707 |

### Agreement by Coding Category
| Column | Description | Mean κ |
|--------|-------------|--------|
| T1 | Threat Model | 0.557 |
| G3 | Application Domain | 0.546 |
| T2 | Transferability | 0.533 |
| G5 | Artifacts Available | 0.469 |
| G6 | Availability Timing | 0.428 |
| G1 | Attack/Defense Type | 0.322 |
| G4 | Publication Venue | 0.321 |
| Q1 | Partial Transferability | 0.300 |
| G2 | Attack Category | 0.256 |

---

## Interpretation (Landis & Koch, 1977)
| κ | Interpretation |
|---|----------------|
| 0.81-1.00 | Almost Perfect |
| 0.61-0.80 | Substantial |
| 0.41-0.60 | Moderate |
| 0.21-0.40 | Fair |
| < 0.21 | Slight/Poor |

---

## Files
| File | Description |
|------|-------------|
| `global_agreement_matrix.csv` | 5×5 κ matrix |
| `per_column_agreement.csv` | Per-category κ for all pairs |
| `reliability_heatmap.png` | Matrix visualization |
| `per_column_heatmaps.png` | 9 category heatmaps |
| `reliability_analysis.py` | Full reproducible script |
