#!/usr/bin/env python3
"""
Claude Sonnet 4.5 paper coding for inter-rater reliability.
Model: claude-sonnet-4-5-20250929 (Sonnet 4.5)
Strict allowed values based on human-verified run1 data.
Usage: python code_papers_sonnet45.py <run_number>
  where run_number is 1, 2, or 3

ROBUST VERSION: Handles rate limits, retries, and checkpoint/resume.
"""

import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

# Load API key from .env
from dotenv import load_dotenv

import anthropic
from anthropic import RateLimitError, APIStatusError, APITimeoutError, APIConnectionError

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
RUN1_CSV = BASE_DIR / "human" / "papers_for_coding_71.csv"

# Get run number from command line (default: 1)
RUN_NUMBER = int(sys.argv[1]) if len(sys.argv) > 1 else 1
if RUN_NUMBER not in [1, 2, 3]:
    print(f"ERROR: Run number must be 1, 2, or 3 (got {RUN_NUMBER})")
    sys.exit(1)

OUTPUT_FILE = Path(__file__).parent / f"run{RUN_NUMBER}.csv"

# Load .env from project root
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=True)

# Initialize Anthropic client with built-in retry
# SDK auto-retries 429, 500+, connection errors with exponential backoff
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_retries=5,  # Increase from default 2
    timeout=120.0,  # 2 min timeout per request
)

# Checkpoint file for resume capability
CHECKPOINT_FILE = Path(__file__).parent / f"checkpoint_run{RUN_NUMBER}.json"

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
        
        # Limit text length (conservatively to ~25k tokens input)
        # At ~4 chars/token, 100k chars = ~25k tokens, well under 30k ITPM limit
        if len(text) > 100000:
            text = text[:100000] + "\n\n[TRUNCATED]"
        
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


# ============================================================================
# RATE LIMIT CONFIGURATION (Tier 1: 50 RPM, 30k ITPM, 8k OTPM)
# ============================================================================
# Strategy: Stay well under limits to avoid 429s
# - 50 RPM = 1.2 seconds minimum between requests
# - 30k ITPM with ~25k tokens/paper = max 1.2 papers/min
# - We use 65s between requests = ~0.92 req/min (safe margin)

BASE_WAIT = 65  # seconds between successful requests (safe for Tier 1)
MIN_WAIT = 30   # minimum wait (for when we have lots of headroom)
MAX_WAIT = 300  # maximum wait (5 min) when severely rate limited
MAX_RETRIES_PER_PAPER = 5  # max retries for a single paper before skipping


def exponential_backoff_with_jitter(attempt, base_delay=60, max_delay=300):
    """Calculate wait time with exponential backoff + jitter to avoid thundering herd."""
    # Exponential: 60, 120, 240, 300, 300... (capped at max)
    delay = min(base_delay * (2 ** attempt), max_delay)
    # Add jitter: +/- 20% randomness
    jitter = delay * 0.2 * (random.random() * 2 - 1)
    return delay + jitter


def get_retry_after(exception):
    """Extract retry-after value from rate limit error if available."""
    try:
        if hasattr(exception, 'response') and exception.response:
            retry_after = exception.response.headers.get('retry-after')
            if retry_after:
                return float(retry_after)
    except (ValueError, AttributeError):
        pass
    return None


def load_checkpoint():
    """Load checkpoint to resume from where we left off."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                data = json.load(f)
                return set(data.get('completed_arxiv_ids', []))
        except (json.JSONDecodeError, IOError):
            pass
    return set()


def save_checkpoint(completed_ids):
    """Save checkpoint of completed paper IDs."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'completed_arxiv_ids': list(completed_ids)}, f)


def code_paper(paper_text, paper_title, arxiv_id):
    """Call Claude Sonnet 4.5 to code a paper with robust error handling."""
    content = None

    for attempt in range(MAX_RETRIES_PER_PAPER):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-5-20250929",  # Sonnet 4.5
                max_tokens=1500,  # Reduced - JSON response doesn't need 2000
                temperature=0.0,  # Zero temperature for consistency
                system=CODING_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": f"Paper Title: {paper_title}\narXiv ID: {arxiv_id}\n\n---\n\n{paper_text}"
                    }
                ]
            )

            content = message.content[0].text.strip()

            # Remove markdown code blocks if present
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'\s*```$', '', content)

            # Parse JSON
            result = json.loads(content)
            return result, None  # Success, no error

        except json.JSONDecodeError as e:
            print(f"\n  JSON parse error: {e}")
            print(f"  Response preview: {content[:200] if content else 'None'}...")
            return None, f"JSON parse error: {e}"

        except RateLimitError as e:
            # SDK already retried 5 times with backoff, this is persistent rate limiting
            retry_after = get_retry_after(e)
            if retry_after:
                wait_time = retry_after + random.uniform(5, 15)  # Add jitter
                print(f"\n  Rate limited (retry-after={retry_after:.0f}s). Waiting {wait_time:.0f}s...")
            else:
                wait_time = exponential_backoff_with_jitter(attempt)
                print(f"\n  Rate limited (attempt {attempt+1}/{MAX_RETRIES_PER_PAPER}). Waiting {wait_time:.0f}s...")

            if attempt < MAX_RETRIES_PER_PAPER - 1:
                time.sleep(wait_time)
                continue
            return None, f"Rate limit exceeded after {MAX_RETRIES_PER_PAPER} retries"

        except APITimeoutError as e:
            wait_time = exponential_backoff_with_jitter(attempt, base_delay=30)
            print(f"\n  Timeout (attempt {attempt+1}/{MAX_RETRIES_PER_PAPER}). Waiting {wait_time:.0f}s...")
            if attempt < MAX_RETRIES_PER_PAPER - 1:
                time.sleep(wait_time)
                continue
            return None, f"Timeout after {MAX_RETRIES_PER_PAPER} retries"

        except APIConnectionError as e:
            wait_time = exponential_backoff_with_jitter(attempt, base_delay=30)
            print(f"\n  Connection error (attempt {attempt+1}/{MAX_RETRIES_PER_PAPER}). Waiting {wait_time:.0f}s...")
            if attempt < MAX_RETRIES_PER_PAPER - 1:
                time.sleep(wait_time)
                continue
            return None, f"Connection error after {MAX_RETRIES_PER_PAPER} retries"

        except APIStatusError as e:
            # 500, 502, 503, 529 (overloaded), etc.
            if e.status_code >= 500 or e.status_code == 529:
                wait_time = exponential_backoff_with_jitter(attempt, base_delay=60)
                print(f"\n  Server error {e.status_code} (attempt {attempt+1}/{MAX_RETRIES_PER_PAPER}). Waiting {wait_time:.0f}s...")
                if attempt < MAX_RETRIES_PER_PAPER - 1:
                    time.sleep(wait_time)
                    continue
            return None, f"API error {e.status_code}: {e.message}"

        except Exception as e:
            # Unexpected error - don't retry
            print(f"\n  Unexpected error: {type(e).__name__}: {e}")
            return None, f"Unexpected error: {e}"

    return None, "Max retries exceeded"


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
    print(f"CLAUDE SONNET 4.5 PAPER CODING - RUN {RUN_NUMBER}/3")
    print("=" * 70)

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not found in .env")
        sys.exit(1)

    # Load paper list (71 papers, same corpus as human coding)
    papers = []
    with open(RUN1_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            papers.append(row)

    print(f"Loaded {len(papers)} papers from {RUN1_CSV.name}")
    print(f"PDFs directory: {PAPERS_DIR}")
    print(f"Output: {OUTPUT_FILE}")

    # Load checkpoint for resume capability
    completed_ids = load_checkpoint()
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} papers already completed")
    print()

    # Create/load output file
    OUTPUT_FILE.parent.mkdir(exist_ok=True)

    # If resuming, load existing results; otherwise start fresh
    coded_papers = []
    if completed_ids and OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                coded_papers.append(row)
        print(f"Loaded {len(coded_papers)} existing results from output file")
    else:
        # Start fresh - write header
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()

    # Process each paper
    errors = []
    validation_errors = []
    papers_processed_this_session = 0

    for i, paper in enumerate(papers):
        arxiv_id = paper['arxiv_id']
        title = paper['paper_title']

        # Skip if already completed (checkpoint resume)
        if arxiv_id in completed_ids:
            continue

        papers_processed_this_session += 1
        remaining = len(papers) - len(completed_ids) - papers_processed_this_session + 1
        eta_minutes = remaining * BASE_WAIT / 60

        print(f"[{i+1}/{len(papers)}] {arxiv_id}: {title[:40]}...", end=" ", flush=True)
        print(f"(~{eta_minutes:.0f}m remaining)", end=" ", flush=True)

        # Find PDF
        pdf_path = find_pdf_for_paper(title, arxiv_id, PAPERS_DIR)

        if not pdf_path:
            print("PDF NOT FOUND")
            errors.append((arxiv_id, "PDF not found"))
            for col in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'T1', 'T2', 'Q1']:
                paper[col] = ''
            coded_papers.append(paper)
            completed_ids.add(arxiv_id)
            save_checkpoint(completed_ids)
            continue

        # Extract text
        text = extract_pdf_text(pdf_path)
        if not text:
            print("TEXT EXTRACTION FAILED")
            errors.append((arxiv_id, "Text extraction failed"))
            for col in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'T1', 'T2', 'Q1']:
                paper[col] = ''
            coded_papers.append(paper)
            completed_ids.add(arxiv_id)
            save_checkpoint(completed_ids)
            continue

        # Code paper with robust error handling
        result, error = code_paper(text, title, arxiv_id)

        if result:
            # Validate coding
            val_errors = validate_coding(result)
            if val_errors:
                print(f"VALIDATION: {'; '.join(val_errors)}")
                validation_errors.append((arxiv_id, val_errors))

            # Update paper with coding
            for col in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'T1', 'T2', 'Q1']:
                paper[col] = result.get(col, '')
            print("OK")
        else:
            print(f"FAILED: {error}")
            errors.append((arxiv_id, error))
            for col in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'T1', 'T2', 'Q1']:
                paper[col] = ''

        coded_papers.append(paper)

        # Write to CSV immediately (real-time progress)
        with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writerow(paper)

        # Save checkpoint
        completed_ids.add(arxiv_id)
        save_checkpoint(completed_ids)

        # Rate limiting with jitter to avoid predictable patterns
        wait_time = BASE_WAIT + random.uniform(-5, 10)
        print(f"  Waiting {wait_time:.0f}s before next request...")
        time.sleep(wait_time)

    # Final save (ensures consistency)
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(coded_papers)

    # Clean up checkpoint on successful completion
    if len(completed_ids) == len(papers):
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            print("Checkpoint file removed (run complete)")

    print()
    print("=" * 70)
    print(f"COMPLETE: {len(papers) - len(errors)}/{len(papers)} papers coded successfully")
    print(f"Output saved to: {OUTPUT_FILE}")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for arxiv_id, err in errors[:10]:
            print(f"  {arxiv_id}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    if validation_errors:
        print(f"\nValidation warnings ({len(validation_errors)}):")
        for arxiv_id, errs in validation_errors[:5]:
            print(f"  {arxiv_id}: {errs}")
    print("=" * 70)
    print(f"\nEstimated cost per run: ~$6 (71 papers, Sonnet 4.5)")
    print(f"Actual papers processed this session: {papers_processed_this_session}")


if __name__ == "__main__":
    main()
