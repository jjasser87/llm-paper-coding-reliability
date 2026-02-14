# Gemini 3 Pro Paper Coding Runs

## Overview
This folder contains the Gemini 3 Pro coding script for evaluating LLM performance on security paper coding.

## Model Details
- **Model**: Gemini 3.0 Pro
- **Cost per run**: $2.22 (71 papers)
- **Total cost (3 runs)**: $6.66
- **Temperature**: 0.0 (for consistency)

## Usage

### Run 1
```bash
python code_papers_gemini3.py
```
Output: `run1.csv`

### Run 2
Edit line 32 in `code_papers_gemini3.py`:
```python
OUTPUT_FILE = Path(__file__).parent / "run2.csv"
```
Then run:
```bash
python code_papers_gemini3.py
```

### Run 3
Edit line 32 in `code_papers_gemini3.py`:
```python
OUTPUT_FILE = Path(__file__).parent / "run3.csv"
```
Then run:
```bash
python code_papers_gemini3.py
```

## Requirements
- Google API key in `.env` file (parent directory) as `GOOGLE_API_KEY`
- Python packages: `google-generativeai`, `pymupdf`, `python-dotenv`

## Output Files
- `run1.csv` - First independent run
- `run2.csv` - Second independent run
- `run3.csv` - Third independent run

All runs use the same prompt and papers for inter-rater reliability analysis.
