# StoryAudit: Backstory Consistency Checker
**KDSH 2026 - Track A: Backstory Consistency with Free Google Technologies**

A fully functional end-to-end system for checking whether character backstories are consistent with long-form narratives, using **ONLY FREE Google APIs and Pathway framework**.

## Quick Start (2 minutes)

```bash
# 1. Install packages
pip install google-generativeai pathway pandas numpy

# 2. Get free Google API key from https://ai.google.dev
# Set environment variable:
export GOOGLE_API_KEY="your-key-here"

# 3. Run on included test data
python run.py --story-id 1

# 4. Check results.csv for predictions
```

## Key Features

✅ **ZERO Cost** - Free Google Gemini API (no credit card)  
✅ **Fully Functional** - End-to-end consistency checking  
✅ **Pathway Integration** - Uses Pathway framework for document processing  
✅ **Deterministic** - Reproducible results, JSON-based reasoning  
✅ **Test Data Included** - Run immediately with sample stories  

## System Architecture

### Pipeline

```
Load Documents
    ↓
Chunk Narrative (Pathway)
    ↓
Extract Claims (Gemini API)
    ↓
Retrieve Evidence (Pathway indexing)
    ↓
Verify Claims (Gemini API)
    ↓
Make Decision (Deterministic rule)
    ↓
Output: results.csv
```

### Decision Rule
- **If ANY high-confidence (≥0.8) contradiction found** → Label = 0 (Inconsistent)
- **Otherwise** → Label = 1 (Consistent)

## Technology Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| **LLM** | Google Gemini API (gemini-1.5-flash) | FREE |
| **Framework** | Pathway (Python) | FREE |
| **Data** | Pandas, NumPy | FREE |
| **Total Cost** | **$0** | ✓ |

## File Structure

```
StoryAudit/
├── run.py                      # Main entry point
├── config.py                   # Config & prompts
├── requirements.txt
├── SETUP.md                    # Detailed setup guide
│
├── src/                        # Core components
│   ├── ingest.py              # Document loading (Pathway)
│   ├── chunk.py               # Narrative chunking
│   ├── claims.py              # Claim extraction (Gemini)
│   ├── retrieve.py            # Evidence retrieval (Pathway)
│   ├── judge.py               # Verification (Gemini)
│   └── pipeline.py            # Orchestration
│
├── data/
│   ├── narratives/            # *.txt - full novels
│   └── backstories/           # *.txt - character backgrounds
│
└── results.csv               # Output file (generated)
```

## Usage

### Basic Usage

**Process single story:**
```bash
python run.py --story-id 1
```

**Process all stories:**
```bash
python run.py --all
```

**With verbose output:**
```bash
python run.py --story-id 1 --verbose
```

**Validate setup:**
```bash
python verify_setup.py
```

**Run pipeline demo:**
```bash
python demo_pipeline.py
```

### Data Format

**Narratives** (in `data/narratives/`):
- Files: `story_1.txt`, `story_2.txt`, etc.
- Format: Plain text (UTF-8)
- Length: 10,000+ words (full novels)
- Content: Complete chronological story

**Backstories** (in `data/backstories/`):
- Files: `backstory_1.txt`, `backstory_2.txt`, etc. (match narrative IDs)
- Format: Plain text (UTF-8)
- Length: 1,000+ words
- Content: Character background claims

### Output Format

`results.csv`:
```
Story ID,Prediction,Rationale
1,1,"No contradictions found (15 claims verified successfully)"
2,0,"Contradiction found: Character was trained in military combat... (confidence: 0.92)"
```

**Prediction**:
- `1` = Consistent backstory
- `0` = Inconsistent backstory

## How It Works

### 1. **Document Loading** (ingest.py)
- Load full narrative and backstory texts
- Validate minimum word counts
- Use Pathway for document management

### 2. **Chunking** (chunk.py)
- Split narrative into overlapping temporal chunks
- Preserve chronological order
- Maintain context across boundaries

### 3. **Claim Extraction** (claims.py + Gemini)
- Extract atomic, testable claims from backstory
- Categorize: events, traits, skills, relationships, constraints
- Filter vague/non-falsifiable claims
- Prioritize high-impact claims

**Example claims:**
- "Character was trained in military combat before age 20"
- "Character's mother died when character was age 9"
- "Character speaks at least 3 languages fluently"

### 4. **Evidence Retrieval** (retrieve.py + Pathway)
- For each claim, find relevant narrative chunks
- Use keyword matching + term weighting
- Consider keyword proximity
- Return top-5 most relevant chunks per claim
- Use Pathway inverted index for fast retrieval

### 5. **Consistency Verification** (judge.py + Gemini)
- Verify each claim against retrieved evidence
- Determine: CONSISTENT vs CONTRADICTION
- Return confidence scores (0.0-1.0)

**Gemini is asked:**
"Does this narrative evidence make the backstory claim impossible?"

**Conservative approach:**
- Absence of evidence ≠ Contradiction
- Ambiguities resolved in favor of CONSISTENT
- Only high-confidence contradictions cause rejection

### 6. **Final Decision** (pipeline.py)
- Apply decision rule:
  - Any high-confidence (≥0.8) contradiction → Label = 0
  - No contradictions → Label = 1
- Write predictions to `results.csv`

## Pathway Integration

Pathway is used for document processing:

```python
# Document ingestion
table = pw.debug.table_from_rows(...)
```

```python
# Evidence indexing
inverted_index = self._build_inverted_index()  # Terms → Chunks
```

```python
# Fast retrieval
candidates = fast_retrieve(query_terms, top_k=5)
```

## API Usage

### Google Gemini API (Free Tier)

**Model:** `gemini-1.5-flash`  
**Temperature:** 0.0 (deterministic)  
**Max Tokens:** 3000 (extraction), 1500 (verification)

**Cost:** $0 - Using free tier with no credit card

**Prompts:**
1. Claim extraction: Extract 10-25 atomic claims from backstory
2. Consistency verification: Does evidence make claim impossible?

Both prompts request JSON output for reliable parsing.

## Configuration

Key settings in `config.py`:

```python
# Chunking
CHUNK_SIZE = 2500              # Words per chunk
CHUNK_OVERLAP = 300            # Word overlap

# Retrieval
TOP_K_CHUNKS = 5               # Chunks per claim
SIMILARITY_THRESHOLD = 0.3     # Minimum relevance

# Verification
CONTRADICTION_CONFIDENCE_THRESHOLD = 0.8
MAX_TOKENS_EXTRACTION = 3000
MAX_TOKENS_VERIFICATION = 1500
```

## Test Data

Sample stories are included:

**Story 1: "The Merchant's Journey"**
