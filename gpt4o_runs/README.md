# GPT-4o Paper Coding Runs

## Overview
This folder contains the GPT-4o coding script for evaluating LLM performance on security paper coding.

## Model Details
- **Model**: GPT-4o
- **Cost per run**: $2.78 (71 papers)
- **Total cost (3 runs)**: $8.34
- **Temperature**: 0.0 (for consistency)

## Usage

### Run 1
```bash
python code_papers_gpt4o.py
```
Output: `run1.csv`

### Run 2
Edit line 33 in `code_papers_gpt4o.py`:
```python
OUTPUT_FILE = Path(__file__).parent / "run2.csv"
```
Then run:
```bash
python code_papers_gpt4o.py
```

### Run 3
Edit line 33 in `code_papers_gpt4o.py`:
```python
OUTPUT_FILE = Path(__file__).parent / "run3.csv"
```
Then run:
```bash
python code_papers_gpt4o.py
```

## Requirements
- OpenAI API key in `.env` file (parent directory)
- Python packages: `openai`, `pymupdf`, `python-dotenv`

## Output Files
- `run1.csv` - First independent run
- `run2.csv` - Second independent run
- `run3.csv` - Third independent run

All runs use the same prompt and papers for inter-rater reliability analysis.
