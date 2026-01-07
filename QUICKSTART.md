# Quick Start Guide

Get up and running in 5 minutes.

## Prerequisites

- Python 3.9+
- Anthropic API key ([Get one here](https://console.anthropic.com))

## Installation

```bash
# 1. Install dependencies
pip install pathway==0.15.0 anthropic pandas numpy

# 2. Set up API key
export ANTHROPIC_API_KEY="your-key-here"

# 3. Create data directories
mkdir -p data/narratives data/backstories
```

## Data Setup

Place your files:

```
data/
├── narratives/
│   └── story_1.txt      # Your novel (100k+ words)
└── backstories/
    └── backstory_1.txt  # Character backstory
```

## Run Your First Test

```bash
# Test single story
python run.py --story-id 1 --verbose

# Check results
cat results.csv
```

Expected output:
```
Story ID,Prediction,Rationale
1,1,"Backstory consistent with narrative (12/15 claims verified)"
```

## What Just Happened?

The system:
1. ✅ Loaded your novel (100k+ words)
2. ✅ Split it into 40-50 chunks (2500 words each)
3. ✅ Extracted 15 testable claims from backstory
4. ✅ Found relevant evidence for each claim
5. ✅ Verified consistency
6. ✅ Made final decision: CONSISTENT (1)

## Common Issues

### "API key not found"
```bash
export ANTHROPIC_API_KEY="your-key"
```

### "Narrative not found"
Make sure filename matches:
- `story_1.txt` or `1.txt` for story ID "1"

### "Processing too slow"
This is normal! Each story takes ~15-20 minutes due to:
- Multiple LLM API calls (15-25 per story)
- Rate limiting
- Complex reasoning

## Next Steps

1. **Process more stories**:
   ```bash
   python run.py --story-ids 1 2 3
   ```

2. **Process all stories**:
   ```bash
   python run.py --all
   ```

3. **Customize configuration**:
   Edit `config.py` to adjust:
   - Chunk size
   - Number of claims to extract
   - Contradiction threshold

4. **Read full documentation**:
   - `README.md` - Complete system documentation
   - `REPORT.md` - Technical deep dive

## Validation

Test your setup:
```bash
python run.py --validate
```

Should show:
```
✓ Environment validation passed
  Narratives dir: /path/to/data/narratives
  Backstories dir: /path/to/data/backstories
  API key: Set
```

## Submission Checklist

Before submitting:

- [ ] All code runs without errors
- [ ] `results.csv` generated with correct format
- [ ] `REPORT.md` completed (max 10 pages)
- [ ] Code is well-commented
- [ ] README explains system clearly
- [ ] Pathway integration is meaningful

## Need Help?

- Check `README.md` for detailed documentation
- Review `REPORT.md` for technical details
- Look at source code comments
- Contact: [Your email/Discord]

---

**Good luck with the hackathon! 🚀**