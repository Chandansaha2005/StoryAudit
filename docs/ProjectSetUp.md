# Kharagpur Data Science Hackathon 2026
## Backstory Consistency Checker - Track A Implementation

### 🎯 Project Overview

This project evaluates whether hypothetical character backstories are logically consistent with long-form narratives (100k+ word novels).

**Goal**: Build a system that can detect global inconsistencies in character backgrounds by analyzing causal relationships, character development, and narrative constraints across entire novels.

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Python 3.9+
python --version

# Install dependencies
pip install pathway-python
pip install anthropic
pip install pandas
pip install numpy
```

### 2. Project Structure

```
kdsh_2026/
├── data/
│   ├── narratives/           # Place .txt files here
│   └── backstories/          # Place backstory files here
├── src/
│   ├── consistency_checker.py
│   ├── pathway_pipeline.py
│   └── utils.py
├── results.csv               # Output file
├── report.pdf               # 10-page report
└── README.md
```

### 3. Setup API Keys

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-key-here"

# Or add to .env file
echo "ANTHROPIC_API_KEY=your-key" > .env
```

---

## 📊 How It Works

### Pipeline Overview

```
1. INGEST → 2. CHUNK → 3. EXTRACT → 4. VERIFY → 5. DECIDE
    ↓           ↓           ↓           ↓           ↓
  Novel      2K words   Backstory    Evidence    Binary
  Text       chunks      Claims      Analysis    Output
```

### Key Components

#### 1. **Document Chunking**
- Splits novels into ~2000-word chunks with 200-word overlap
- Preserves sentence/paragraph boundaries
- Maintains context across chunks

#### 2. **Claim Extraction**
- Identifies testable assertions in backstory
- Categories: events, skills, traits, relationships, beliefs
- Extracts 10-20 key claims per backstory

#### 3. **Evidence Retrieval**
- Finds relevant narrative passages for each claim
- Uses keyword matching + semantic similarity
- Retrieves top 3-5 most relevant chunks per claim

#### 4. **Consistency Verification**
- Checks each claim against narrative evidence
- Detects: contradictions, causal impossibilities, character mismatches
- Classifies: SUPPORTED, NEUTRAL, or CONTRADICTED

#### 5. **Decision Aggregation**
- Combines all verification results
- Threshold: >20% contradictions → INCONSISTENT
- Generates evidence-based rationale

---

## 🎓 Strategy Tips for Qualifying

### What Judges Look For (Track A)

1. **Correctness** (40%)
   - Accurate binary classifications
   - Low false positive/negative rate
   - Robust to edge cases

2. **Novelty** (30%)
   - Beyond basic RAG templates
   - Creative evidence aggregation
   - Multi-stage reasoning

3. **Long Context Handling** (30%)
   - Effective chunking strategy
   - Global consistency tracking
   - Memory mechanisms

### Quick Wins

✅ **Do This**:
- Use overlapping chunks to preserve context
- Extract structured claims (not just summarize)
- Verify multiple types of consistency (causal, character, world)
- Provide specific evidence quotes in rationale
- Test on edge cases

❌ **Avoid This**:
- Simple keyword matching alone
- Single-pass classification without verification
- Ignoring causal relationships
- Generic/vague rationales
- Over-relying on one model call

---

## 💡 Implementation Examples

### Basic Usage

```python
from consistency_checker import PathwayBackstoryChecker

# Initialize
checker = PathwayBackstoryChecker(api_key="your-key")

# Evaluate single story
prediction, rationale = checker.evaluate(
    narrative_path="data/narratives/story_1.txt",
    backstory="Character grew up in military family..."
)

print(f"Result: {prediction}")  # 1 or 0
print(f"Why: {rationale}")
```

### Batch Processing

```python
import pandas as pd
import os

results = []

for story_id in range(1, 11):  # Process 10 stories
    narrative_path = f"data/narratives/story_{story_id}.txt"
    backstory_path = f"data/backstories/backstory_{story_id}.txt"
    
    with open(backstory_path, 'r') as f:
        backstory = f.read()
    
    prediction, rationale = checker.evaluate(narrative_path, backstory)
    
    results.append({
        'Story ID': story_id,
        'Prediction': prediction,
        'Rationale': rationale
    })

# Save results
df = pd.DataFrame(results)
df.to_csv('results.csv', index=False)
```

---

## 🔧 Advanced Features

### 1. Enhanced Retrieval

```python
# Use embeddings for better chunk retrieval
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_search(query, chunks, k=5):
    query_emb = model.encode([query])
    chunk_embs = model.encode(chunks)
    
    # Cosine similarity
    scores = (query_emb @ chunk_embs.T)[0]
    top_indices = scores.argsort()[-k:][::-1]
    
    return [chunks[i] for i in top_indices]
```

### 2. Multi-Stage Verification

```python
def verify_with_confirmation(claim, evidence):
    # Stage 1: Initial check
    initial = check_consistency(claim, evidence)
    
    # Stage 2: If contradiction found, double-check
    if initial == "CONTRADICTED":
        confirmation = recheck_contradiction(claim, evidence)
        return confirmation
    
    return initial
```

### 3. Causal Chain Analysis

```python
def analyze_causal_chain(backstory_event, future_events):
    """
    Check if backstory event makes future events possible.
    """
    prompt = f"""
    Given this backstory event: {backstory_event}
    And these future events: {future_events}
    
    Does the backstory event:
    1. Enable these future events?
    2. Make them impossible?
    3. Have no causal relationship?
    """
    # ... implementation
```

---

## 📝 Report Guidelines

Your 10-page report should cover:

1. **Introduction** (1 page)
   - Problem understanding
   - Your approach overview

2. **System Design** (3 pages)
   - Architecture diagram
   - Component descriptions
   - Pathway integration details

3. **Long Context Handling** (2 pages)
   - Chunking strategy
   - Memory mechanisms
   - Global consistency tracking

4. **Evaluation** (2 pages)
   - Results on test set
   - Performance metrics
   - Example cases

5. **Limitations & Future Work** (2 pages)
   - Known failure modes
   - Ideas for improvement
   - Lessons learned

---

## 🎯 Qualification Checklist

- [ ] Pathway integrated for document management
- [ ] Handles 100k+ word narratives
- [ ] Extracts structured backstory claims
- [ ] Retrieves relevant evidence passages
- [ ] Multi-step consistency verification
- [ ] Aggregates evidence for final decision
- [ ] Provides specific rationales
- [ ] Results.csv with correct format
- [ ] Code is reproducible
- [ ] Report explains approach clearly

---

## 📚 Resources

**Pathway**:
- [Documentation](https://pathway.com/developers/documentation)
- [LLM Integration](https://pathway.com/developers/documentation/llm-xpack)
- [Vector Store](https://pathway.com/developers/documentation/vector-store)

**Anthropic Claude**:
- [API Docs](https://docs.anthropic.com)
- [Prompt Engineering](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering)

**Competition**:
- [Whatsapp Community](link-from-doc)
- [Discord Channel](link-from-doc)

---

## 🏆 Pro Tips

1. **Start Simple**: Get a basic pipeline working first, then enhance
2. **Test Incrementally**: Verify each component independently
3. **Use Examples**: Study the provided examples carefully
4. **Ask Questions**: Use Discord/Whatsapp for clarifications
5. **Time Management**: Leave time for report writing
6. **Document Everything**: Keep notes for your report

---

## 🐛 Common Issues

**Issue**: Out of memory with large novels
**Fix**: Increase chunk size, reduce context per LLM call

**Issue**: Slow processing
**Fix**: Batch claims, limit top-k retrievals, cache embeddings

**Issue**: Generic rationales
**Fix**: Include specific quotes and page references

**Issue**: Pathway integration unclear
**Fix**: Use Pathway for ingestion + indexing, not necessarily for LLM calls

---

## 📧 Support

- Competition Discord: [link]
- Email: [hackathon-email]
- GitHub Issues: [your-repo]

---

Good luck! 🚀 Remember: **clarity > complexity**. A well-explained simple system beats an unexplained complex one.