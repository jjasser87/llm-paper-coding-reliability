# Claude Sonnet 4.5 Paper Coding Runs

## Overview
This folder contains the Claude Sonnet 4.5 coding script for evaluating LLM performance on security paper coding.

## Model Details
- **Model**: Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)
- **Cost per run**: ~$6 (71 papers with increased text limit)
- **Total cost (3 runs)**: ~$18
- **Temperature**: 0.0 (for consistency)
- **Released**: September 29, 2025

## Usage

The script accepts a run number (1, 2, or 3) as a command-line argument.

### Run 1
```bash
python code_papers_sonnet45.py 1
```
Output: `run1.csv`

### Run 2
```bash
python code_papers_sonnet45.py 2
```
Output: `run2.csv`

### Run 3
```bash
python code_papers_sonnet45.py 3
```
Output: `run3.csv`

## Requirements
- Anthropic API key in `.env` file (parent directory)
- Python packages: `anthropic`, `pymupdf`, `python-dotenv`

## Output Files
- `run1.csv` - First independent run
- `run2.csv` - Second independent run
- `run3.csv` - Third independent run

All runs use the same prompt and papers for inter-rater reliability analysis.
