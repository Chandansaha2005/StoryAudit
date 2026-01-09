# StoryAudit - Google Gemini API Setup Guide

## Quick Start

The StoryAudit project has been converted from Anthropic's Claude API to Google Gemini API. Follow these steps to run it with your Gemini API key.

### 1. Get Your Gemini API Key

Visit [Google AI Studio](https://aistudio.google.com/app/apikey) to create or retrieve your API key.

### 2. Set Environment Variable

Before running the project, set your Gemini API key:

**PowerShell:**
```powershell
$env:GEMINI_API_KEY="your-actual-gemini-api-key-here"
cd d:\Projects\StoryAudit
.\.venv\Scripts\python.exe run.py --all
```

**Windows Command Prompt:**
```cmd
set GEMINI_API_KEY=your-actual-gemini-api-key-here
cd d:\Projects\StoryAudit
.venv\Scripts\python.exe run.py --all
```

**Linux/macOS:**
```bash
export GEMINI_API_KEY="your-actual-gemini-api-key-here"
cd ~/Projects/StoryAudit
python run.py --all
```

### 3. Verify Installation

Test that everything is set up correctly:

```bash
python run.py --validate
```

You should see:
```
✓ Environment validation passed
  Narratives dir: D:\Projects\StoryAudit\data\narratives
  Backstories dir: D:\Projects\StoryAudit\data\backstories
  API key: Set
```

### 4. Run the Pipeline

Process all stories:
```bash
python run.py --all
```

Process a single story:
```bash
python run.py --story-id story
```

Process multiple stories:
```bash
python run.py --story-ids story1 story2 story3
```

### 5. View Results

Results are saved to `results.csv` with:
- Story ID
- Prediction (1=consistent, 0=inconsistent)
- Rationale explaining the decision

## Configuration

Key changes from Anthropic to Gemini:

- **Model**: Changed from `claude-sonnet-4-20250514` to `gemini-2.0-flash`
- **API Key Env Var**: `GEMINI_API_KEY` (instead of `ANTHROPIC_API_KEY`)
- **SDK**: Using `google-generativeai` library

All configuration is in `config.py`:
```python
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-2.0-flash"
```

## Requirements

The updated `requirements.txt` includes:
- `google-generativeai>=0.8.0` - Google Gemini API SDK
- `pathway==0.15.0` - For document processing
- `pandas>=2.0.0` - For results handling
- `python-dotenv>=1.0.0` - For environment configuration

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Troubleshooting

**API Key Error:**
- Ensure `GEMINI_API_KEY` is set in environment variables
- Run `python run.py --validate` to check setup

**Import Errors:**
- Verify venv is activated
- Reinstall packages: `pip install -r requirements.txt`

**No results file created:**
- Check data files exist: `data/narratives/` and `data/backstories/`
- Ensure files have .txt extension

## Project Structure

```
StoryAudit/
├── run.py              # Main entry point
├── config.py           # Configuration (now Gemini-based)
├── requirements.txt    # Dependencies with google-generativeai
├── results.csv         # Output file
├── src/
│   ├── claims.py       # Claim extraction (updated for Gemini)
│   ├── chunk.py        # Narrative chunking
│   ├── ingest.py       # Document loading
│   ├── judge.py        # Consistency checking
│   ├── pipeline.py     # Pipeline orchestration
│   └── retrieve.py     # Evidence retrieval
└── data/
    ├── narratives/
    │   └── story.txt   # Your narrative files
    └── backstories/
        └── story.txt   # Your backstory files
```

## Credits

- **Hackathon**: Kharagpur Data Science Hackathon 2026
- **Track**: A (Systems Reasoning with NLP and Generative AI)
- **API**: Google Gemini (formerly Bard)
