#!/usr/bin/env python3
"""
GPT-5.2 paper coding for inter-rater reliability.
Strict allowed values based on human-verified run1 data.
Change OUTPUT_FILE to run1.csv, run2.csv, or run3.csv as needed.
"""

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

# Load API key from .env
from dotenv import load_dotenv

import openai
from openai import OpenAI

# Try to import PDF reader
try:
    import fitz  # PyMuPDF
except ImportError:
    print("Installing PyMuPDF...")
    os.system("pip install pymupdf")
    import fitz

# Configuration
BASE_DIR = Path(__file__).parent.parent
PAPERS_DIR = BASE_DIR / "papers_for_coding"
RUN1_CSV = BASE_DIR / "run1" / "papers_coded_verified_fixed.csv"

# Get run number from command line (default: 1)
RUN_NUMBER = int(sys.argv[1]) if len(sys.argv) > 1 else 1
if RUN_NUMBER not in [1, 2, 3]:
    print(f"ERROR: Run number must be 1, 2, or 3 (got {RUN_NUMBER})")
    sys.exit(1)

OUTPUT_FILE = Path(__file__).parent / f"run{RUN_NUMBER}.csv"

# Load .env from project root
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=True)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Coding prompt with STRICT allowed values from human-verified run1
CODING_PROMPT = """You are coding academic papers for adversarial machine learning research.

For each paper, assign values to exactly 9 variables. Use ONLY the allowed values listed below.

## CODING VARIABLES (STRICT VALUES)

### G1: Paper Type
ALLOWED: "Attack", "Defense", "Evaluation", "Both", "Attack/Defense", "Attack/Evaluation", "Defense/Evaluation", "Attack/Defense/Evaluation"
- Attack: Proposes new attack method
- Defense: Proposes defense/robustness method
- Evaluation: Evaluates existing attacks/defenses
- Combinations: Paper does multiple things (use "/" separator)

### G2: Threat Category
ALLOWED: "Evasion", "Poisoning", "Privacy", "Defense", "N/A"
- Evasion: Fool model at test time (adversarial examples)
- Poisoning: Corrupt training data or model
- Privacy: Steal data/model (membership inference, model extraction)
- Defense: If paper is defense-focused, use "Defense" (not "N/A")
- N/A: Only for pure evaluation papers

### G3: Domain
ALLOWED: "Vision", "NLP", "LLM", "Audio", "Malware", "Tabular", "Cross", "Cross-domain"
- Vision: Image classification (CIFAR, ImageNet)
- NLP: Text classification, sentiment, NER (pre-LLM era)
- LLM: Large language models (GPT, Claude, jailbreaking, prompt injection)
- Audio: Speech recognition, audio attacks
- Malware: Malware detection
- Tabular: Tabular/structured data
- Cross or Cross-domain: 2+ distinct domains

### G4: Publication Venue
ALLOWED: "ML", "Security", "Journal", "arXiv-only"
- ML: NeurIPS, ICML, ICLR, CVPR, ECCV, ICCV, ACL, EMNLP, NAACL
- Security: IEEE S&P, ACM CCS, USENIX Security, NDSS
- Journal: TPAMI, TIFS, TDSC, or other journals
- arXiv-only: No peer-reviewed venue

### G5: Code Available
ALLOWED: "Yes", "No"
- Yes: Public code repository exists (GitHub)
- No: No code available

### G6: Code Release Timing
ALLOWED: "At-pub", "Post-pub", "Never"
- At-pub: Released within 1 month of paper
- Post-pub: Released later than 1 month
- Never: No code released

### T1: Model Access Level (attacks only)
ALLOWED: "White", "Black", "Gray", "White/Black", "N/A"
- White: Has weights, gradients
- Black: Query-only access
- Gray: Surrogate/substitute model
- White/Black: Paper covers both scenarios
- N/A: Defense or evaluation papers

### T2: Gradient Required (attacks only)
ALLOWED: "Yes", "No", "N/A"
- Yes: Uses backpropagation, computes ∇L
- No: Gradient-free (evolutionary, random search, GCG-style)
- N/A: Defense or evaluation papers

### Q1: Real-World Evaluation
ALLOWED: "Yes", "No", "Partial", "N/A"
- Yes: Production system (Google API, Tesla, ChatGPT)
- Partial: Realistic simulation or industry dataset
- No: Standard benchmarks only (CIFAR, ImageNet)
- N/A: Rarely used (check context)

## OUTPUT FORMAT

Return ONLY a valid JSON object (no markdown, no extra text):
{
    "G1": "<one of allowed values>",
    "G2": "<one of allowed values>",
    "G3": "<one of allowed values>",
    "G4": "<one of allowed values>",
    "G5": "<one of allowed values>",
    "G6": "<one of allowed values>",
    "T1": "<one of allowed values>",
    "T2": "<one of allowed values>",
    "Q1": "<one of allowed values>",
    "reasoning": "Brief explanation of key decisions"
}

CRITICAL: Use ONLY the allowed values listed above. Do NOT create new values.
"""


def extract_pdf_text(pdf_path, max_pages=15):
    """Extract text from PDF, limiting to first N pages."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            text += page.get_text()
        doc.close()
        
        # Limit text length
        if len(text) > 50000:
            text = text[:50000] + "\n\n[TRUNCATED]"
        
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None


def find_pdf_for_paper(paper_title, arxiv_id, papers_dir):
    """Find PDF by title or arXiv ID."""
    # Try arxiv_id.pdf first
    pdf_path = papers_dir / f"{arxiv_id}.pdf"
    if pdf_path.exists():
        return pdf_path
    
    # Try with v suffix
    for pdf_file in papers_dir.glob(f"{arxiv_id}*.pdf"):
        return pdf_file
    
    # Try matching by title (PDFs are named by title)
    clean_title = re.sub(r'[^\w\s]', '', paper_title.lower())
    
    for pdf_file in papers_dir.glob("*.pdf"):
        clean_filename = re.sub(r'[^\w\s]', '', pdf_file.stem.lower())
        
        # Check if significant overlap
        title_words = set(clean_title.split())
        file_words = set(clean_filename.split())
        
        if len(title_words & file_words) >= min(4, len(title_words) - 1):
            return pdf_file
    
    return None


def code_paper(paper_text, paper_title, arxiv_id):
    """Call GPT-5.2 to code a paper."""
    try:
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": CODING_PROMPT},
                {"role": "user", "content": f"Paper Title: {paper_title}\narXiv ID: {arxiv_id}\n\n---\n\n{paper_text}"}
            ],
            temperature=0.0,  # Zero temperature for consistency
            max_completion_tokens=1000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        
        # Parse JSON
        result = json.loads(content)
        return result
            
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Response: {content[:200]}")
        return None
    except Exception as e:
        print(f"API error: {e}")
        return None


def validate_coding(result):
    """Validate that all values are from allowed sets."""
    allowed = {
        'G1': {'Attack', 'Defense', 'Evaluation', 'Both', 'Attack/Defense', 'Attack/Evaluation', 
               'Defense/Evaluation', 'Attack/Defense/Evaluation'},
        'G2': {'Evasion', 'Poisoning', 'Privacy', 'Defense', 'N/A'},
        'G3': {'Vision', 'NLP', 'LLM', 'Audio', 'Malware', 'Tabular', 'Cross', 'Cross-domain'},
        'G4': {'ML', 'Security', 'Journal', 'arXiv-only'},
        'G5': {'Yes', 'No'},
        'G6': {'At-pub', 'Post-pub', 'Never'},
        'T1': {'White', 'Black', 'Gray', 'White/Black', 'N/A'},
        'T2': {'Yes', 'No', 'N/A'},
        'Q1': {'Yes', 'No', 'Partial', 'N/A'}
    }
    
    errors = []
    for var, allowed_vals in allowed.items():
        if var in result and result[var] not in allowed_vals:
            errors.append(f"{var}={result[var]} not in {allowed_vals}")
    
    return errors


def main():
    print("=" * 70)
    print(f"GPT-5.2 PAPER CODING - RUN {RUN_NUMBER}/3")
    print("=" * 70)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in .env")
        sys.exit(1)
    
    # Load papers from run1 (same papers)
    papers = []
    with open(RUN1_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            papers.append(row)
    
    print(f"Loaded {len(papers)} papers from run1")
    print(f"PDFs directory: {PAPERS_DIR}")
    print(f"Output: {OUTPUT_FILE}")
    print()
    
    # Process each paper
    coded_papers = []
    errors = []
    validation_errors = []
    
    for i, paper in enumerate(papers):
        arxiv_id = paper['arxiv_id']
        title = paper['paper_title']
        
        print(f"[{i+1}/{len(papers)}] {arxiv_id}: {title[:40]}...", end=" ", flush=True)
        
        # Find PDF
        pdf_path = find_pdf_for_paper(title, arxiv_id, PAPERS_DIR)
        
        if not pdf_path:
            print("PDF NOT FOUND")
            errors.append(arxiv_id)
            # Keep existing values from run1 (blank G1-Q1)
            for col in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'T1', 'T2', 'Q1']:
                paper[col] = ''
            coded_papers.append(paper)
            continue
        
        # Extract text
        text = extract_pdf_text(pdf_path)
        if not text:
            print("TEXT EXTRACTION FAILED")
            errors.append(arxiv_id)
            for col in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'T1', 'T2', 'Q1']:
                paper[col] = ''
            coded_papers.append(paper)
            continue
        
        # Code paper
        result = code_paper(text, title, arxiv_id)
        
        if result:
            # Validate coding
            val_errors = validate_coding(result)
            if val_errors:
                print(f"VALIDATION ERROR: {'; '.join(val_errors)}")
                validation_errors.append((arxiv_id, val_errors))
            
            # Update paper with coding
            for col in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'T1', 'T2', 'Q1']:
                paper[col] = result.get(col, '')
            print("OK")
        else:
            print("CODING FAILED")
            errors.append(arxiv_id)
            for col in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'T1', 'T2', 'Q1']:
                paper[col] = ''
        
        coded_papers.append(paper)
        
        # Rate limiting
        time.sleep(1.5)  # Be nice to OpenAI
    
    # Save results
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(coded_papers)
    
    print()
    print("=" * 70)
    print(f"COMPLETE: {len(papers) - len(errors)}/{len(papers)} papers coded")
    print(f"Output saved to: {OUTPUT_FILE}")
    if errors:
        print(f"\nErrors ({len(errors)}): {', '.join(errors)}")
    if validation_errors:
        print(f"\nValidation errors ({len(validation_errors)}):")
        for arxiv_id, errs in validation_errors[:5]:
            print(f"  {arxiv_id}: {errs}")
    print("=" * 70)
    print(f"\nEstimated cost per run: $2.38 (71 papers with GPT-5.2)")


if __name__ == "__main__":
    main()
