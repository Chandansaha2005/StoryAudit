# KDSH 2026 Track A: Backstory Consistency Checker
## Complete System Implementation

## Project Structure

```
kdsh_track_a/
├── README.md
├── requirements.txt
├── config.py
├── data/
│   ├── narratives/          # Place novel .txt files here
│   └── backstories/         # Place backstory .txt files here
├── src/
│   ├── __init__.py
│   ├── ingest.py           # Pathway-based data loading
│   ├── chunk.py            # Intelligent chunking with temporal ordering
│   ├── claims.py           # Backstory claim extraction
│   ├── retrieve.py         # Evidence retrieval via Pathway
│   ├── judge.py            # Consistency checking logic
│   └── pipeline.py         # End-to-end orchestration
├── run.py                   # Main entry point
├── results.csv             # Output predictions
└── REPORT.md               # Technical report
```

## System Design Principles

### 1. **Long-Context Handling**
- Ordered chunking preserves temporal dependencies
- Overlapping windows maintain cross-chunk context
- Pathway's streaming handles 100k+ words efficiently

### 2. **Constraint-Based Reasoning**
- Backstories decomposed into atomic, testable claims
- Each claim verified against narrative evidence
- Hard contradictions immediately fail (strict logic)

### 3. **Causal Tracking**
- Events ordered temporally within narrative
- Backstory claims checked for causal compatibility
- Future events must remain possible given backstory

### 4. **Evidence Aggregation**
- Multiple passages evaluated per claim
- Contradiction detection uses high-confidence threshold
- Ambiguous cases default to consistent (conservative)

## Files Overview

### Core Implementation Files

1. **config.py** - System configuration and prompts
2. **ingest.py** - Pathway integration for document loading
3. **chunk.py** - Temporal-aware chunking strategy
4. **claims.py** - LLM-based claim extraction
5. **retrieve.py** - Evidence retrieval using Pathway vector store
6. **judge.py** - Consistency verification logic
7. **pipeline.py** - End-to-end orchestration
8. **run.py** - CLI interface for batch processing

Each file is production-ready with error handling, logging, and type hints.