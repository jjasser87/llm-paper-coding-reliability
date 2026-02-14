# LLM Paper Coding Reliability Study

## Overview
Evaluating LLM reliability for coding academic papers in adversarial machine learning (AML) research. We compare four LLMs and one human expert on a fixed corpus of 71 papers and 9 coding variables, with multiple independent runs per model where applicable.

## Methodology (short)

- **Corpus:** 71 AML papers (same ordered list for all raters); LLMs receive full-text from PDFs (PyMuPDF).
- **Coding:** 9 categorical variables (G1–G6: paper type, threat, domain, venue, code availability/timing; T1–T2: attack-related; Q1: real-world evaluation). Allowed values are fixed and shared across LLM prompts.
- **Raters and runs:**

| Rater | Runs | Consensus |
|-------|------|-----------|
| GPT-4o | 3 | Majority vote |
| GPT-5.2 | 3 | Majority vote |
| Gemini-3 | 3 | Majority vote |
| Claude Sonnet 4.5 | 1 | N/A |
| Human | 1 (verified) | N/A |

- **Settings:** Temperature = 0 for all LLM runs; same prompt and paper list per model; independent runs written to `run1.csv` / `run2.csv` / `run3.csv` per folder.
- **Analysis:** Normalization (lowercase, synonym folding), then Fleiss’ κ (intra-rater, 3-run models) and Cohen’s κ (inter-rater). Global matrix merges all 9 columns; per-column κ in `analysis/per_column_agreement.csv`.

**Full methodology (for paper):** See **`analysis/README.md`** — corpus, coding schema table, run details, consensus rule, and statistics are described there in full.

## Data
- **Papers:** 71 AML papers (arXiv IDs and metadata in `human/papers_for_coding_71.csv`; human reference coding in `human/papers_coded_verified_fixed.csv`).
- **Coding categories:** 9 (G1–G6, T1–T2, Q1). Definitions and allowed values are in `analysis/README.md` and in each `code_papers_*.py` prompt.
- **Raters:** GPT-4o, GPT-5.2, Gemini-3, Claude Sonnet 4.5, Human.

## Structure
```
coding/
├── gpt4o_runs/      # 3 runs (run1, run2, run3)
├── gpt52_runs/      # 3 runs
├── gemini3_runs/    # 3 runs
├── sonnet45_runs/   # 1 run
├── human/           # 1 verified run + coding_corrections.csv
└── analysis/        # Methodology, reliability script, results ← START HERE
```

## Results

See **`analysis/README.md`** for the full methodology (suitable for a paper), agreement matrix, per-column κ, and interpretation.

### Summary
| Metric | Value |
|--------|-------|
| Best inter-rater (GPT-5.2 ↔ Gemini-3) | κ = 0.846 |
| Best intra-rater (Gemini-3) | κ = 1.000 |
| Best model–human agreement (Gemini-3) | κ = 0.707 |
