# StoryAudit: KDSH 2026 Track A - Backstory Consistency Checker

## Overview

StoryAudit is a complete end-to-end system for checking the consistency of character backstories against narrative text using only FREE Google technologies.

**Technology Stack:**
- **LLM**: Google Gemini API (free tier, gemini-1.5-flash model)
- **Framework**: Pathway Python framework (for document ingestion and indexing)
- **Language**: Python 3.10+
- **Cost**: ZERO (all APIs are free tier with no credit card requirements)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `google-generativeai>=0.3.0` - Google Gemini API
- `pathway>=0.15.0` - Document processing framework
- `pandas>=2.0.0` - Data output
- `numpy>=1.24.0` - Numerical operations
- `python-dotenv>=1.0.0` - Environment variable management

### 2. Set API Key

Get a free Google API key from: https://ai.google.dev

Set the environment variable:

```bash
# On Windows (PowerShell)
$env:GOOGLE_API_KEY="your-api-key-here"

# On Linux/macOS
export GOOGLE_API_KEY="your-api-key-here"
```

### 3. Run the System

Process a single story:
```bash
python run.py --story-id 1
```

Process multiple stories:
```bash
python run.py --story-ids 1 2 3
```

Process all stories in data directory:
```bash
python run.py --all
```

With detailed logging:
```bash
python run.py --story-id 1 --verbose
```

### 4. Output

Results are saved to `results.csv` with columns:
- `Story ID`: Identifier of the story processed
- `Prediction`: 1 (consistent backstory) or 0 (inconsistent backstory)
- `Rationale`: Explanation of the decision

## System Architecture

### Pipeline Stages

```
1. LOAD DOCUMENTS
   ↓ (ingest.py)
   ├─ Load narrative (full novel text)
   └─ Load backstory (character background)

2. CHUNK NARRATIVE
   ↓ (chunk.py)
   ├─ Detect chapters and temporal boundaries
   └─ Create overlapping chunks with ordering

3. EXTRACT CLAIMS
   ↓ (claims.py + Gemini API)
   ├─ Use LLM to decompose backstory into testable claims
   ├─ Categorize by type (events, traits, skills, etc.)
   └─ Filter to only high-confidence, specific claims

4. RETRIEVE EVIDENCE
   ↓ (retrieve.py + Pathway)
   ├─ For each claim, find relevant narrative chunks
   ├─ Use inverted index for fast retrieval
   └─ Return top-k most relevant chunks

5. VERIFY CLAIMS
   ↓ (judge.py + Gemini API)
   ├─ For each claim, analyze against evidence
   ├─ Use LLM to determine: CONSISTENT vs CONTRADICTION
   └─ Return confidence scores

6. MAKE FINAL DECISION
   ↓ (DecisionAggregator)
   ├─ Apply decision rule:
   │  - If ANY high-confidence contradiction → label = 0
   │  - Otherwise → label = 1
   └─ Write results.csv
```

## File Structure

```
StoryAudit/
├── run.py                    # Main entry point
├── config.py                 # Configuration & prompts
├── requirements.txt          # Python dependencies
├── results.csv              # Output file (generated)
│
├── src/
│   ├── __init__.py
│   ├── ingest.py            # Document loading (Pathway-based)
│   ├── chunk.py             # Narrative chunking
│   ├── claims.py            # Claim extraction (Gemini API)
│   ├── retrieve.py          # Evidence retrieval (Pathway indexing)
│   ├── judge.py             # Consistency verification (Gemini API)
│   └── pipeline.py          # Orchestration
│
└── data/
    ├── narratives/          # *.txt files - full novels
    └── backstories/         # *.txt files - character backgrounds
```

## Input Format

### Narrative Files
- Location: `data/narratives/`
- Naming: `story_1.txt`, `story_2.txt`, etc.
- Format: Plain text files containing the complete novel/narrative
- Length: Typically 10,000+ words
- Content: Full chronological story to be checked

### Backstory Files
- Location: `data/backstories/`
- Naming: `backstory_1.txt`, `backstory_2.txt`, etc. (must match narrative IDs)
- Format: Plain text file describing character background
- Length: Typically 1,000+ words
- Content: Factual backstory claims about character

## How It Works

### 1. Claim Extraction (using Gemini)
The system uses a prompt to decompose backstories into specific, testable claims:

Example claims extracted:
- "Character was trained in military combat before age 20"
- "Character's mother died when character was age 9"
- "Character speaks at least 3 languages fluently"
- "Character experienced parental abandonment"

### 2. Evidence Retrieval (using Pathway)
For each claim, the system retrieves the most relevant chunks from the narrative:
- Uses keyword matching and term weighting
- Considers proximity of keywords
- Returns top 5 chunks in temporal order

### 3. Consistency Checking (using Gemini)
Gemini analyzes each claim against the retrieved evidence:

**CONSISTENT**: Narrative supports or is neutral regarding the claim
**CONTRADICTION**: Narrative directly contradicts or makes claim impossible

Confidence scores indicate the certainty of each verdict.

### 4. Final Decision
- **If ANY high-confidence contradiction found**: Label = 0 (backstory inconsistent)
- **Otherwise**: Label = 1 (backstory consistent with narrative)

This strict rule ensures false positives (incorrectly accepting inconsistent backstories) are minimized.

## Key Features

### Pathway Integration (MANDATORY)
The system uses Pathway for document management and retrieval:
- `DocumentIngestion`: Loads and validates documents
- `PathwayDocumentStore`: Manages documents in Pathway tables
- `PathwayEvidenceIndex`: Builds inverted index for fast chunk retrieval
- `retrieve.py`: Uses inverted index for evidence retrieval

### Gemini API Usage (FREE TIER)
- Model: `gemini-1.5-flash` (optimized for speed and cost)
- Temperature: 0.0 (deterministic outputs)
- Input: Short, specific prompts requesting JSON output
- No system instructions or complex chaining

### Robustness
- Graceful handling of API errors (defaults to CONSISTENT if verification fails)
- Absence of evidence ≠ contradiction (conservative approach)
- Proper sentence validation (low-confidence results given less weight)

## Testing

Test data is provided in:
- `data/narratives/story_1.txt` - The Merchant's Journey
- `data/backstories/backstory_1.txt` - Marcus's backstory
- `data/narratives/story_2.txt` - The Scholar's Quest  
- `data/backstories/backstory_2.txt` - Eleanor's backstory

Run a quick test:
```bash
python run.py --story-id 1 --verbose
```

Expected output: `results.csv` with predictions for story_1

## Limitations & Notes

### Environment Variables
- Must set `GOOGLE_API_KEY` before running
- No credit card required; free tier API key sufficient

### File Requirements
- Narrative and backstory files must exist and be readable UTF-8
- Filename patterns: `story_N.txt` and `backstory_N.txt` where N is story ID
- Minimum 100 words for backstory, 10,000 words for narrative

### Language
- All prompts are in English
- System can process narratives in English (other languages possible but untested)

### Accuracy
- System is deterministic (same inputs produce same outputs)
- Accuracy depends on:
  - Quality of backstory and narrative
  - Clarity of claims extracted
  - Appropriateness of evidence chunks retrieved

## Configuration

Key settings in `config.py`:
```python
CHUNK_SIZE = 2500              # Words per narrative chunk
CHUNK_OVERLAP = 300            # Word overlap between chunks
TOP_K_CHUNKS = 5               # Chunks to retrieve per claim
CONTRADICTION_CONFIDENCE_THRESHOLD = 0.8  # Confidence threshold for contradiction
MAX_TOKENS_EXTRACTION = 3000   # Max tokens for claim extraction
MAX_TOKENS_VERIFICATION = 1500 # Max tokens for verification
```

## Troubleshooting

**"GOOGLE_API_KEY not set in environment"**
- Solution: Set the environment variable before running

**"No narrative files found"**
- Solution: Ensure `.txt` files exist in `data/narratives/`

**"Verification error"**
- The system defaults to CONSISTENT if Gemini API fails
- Check your internet connection and API key validity

**"No JSON found in response"**
- Gemini sometimes returns non-JSON. System has fallback logic.
- Rare but can happen; re-run to retry

## Performance

Typical execution times:
- Single story: 2-5 minutes (depending on narrative length)
- 10 stories: 20-50 minutes
- API calls: 1 extraction call + N verification calls per story

No cost for any API calls (free tier Gemini API).

## Output Format

`results.csv`:
```
Story ID,Prediction,Rationale
1,1,"No contradictions found (15 claims verified successfully)"
2,0,"Contradiction found: Character was trained in military combat before age 20... (confidence: 0.92)"
```

Predictions:
- **1** = Consistent backstory (accepted)
- **0** = Inconsistent backstory (rejected)

## License & Attribution

This project was developed for Kharagpur Data Science Hackathon (KDSH) 2026 - Track A.
All code uses only FREE, open-source, and free-tier API services.

## Contact & Support

For issues or questions, refer to the inline code documentation and this README.

---

**Remember**: This system requires an active internet connection and a valid Google API key. All processing uses the free tier of Google Gemini API with no credit card required.
