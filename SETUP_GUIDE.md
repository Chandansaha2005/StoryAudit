# StoryAudit Setup Guide

Complete step-by-step instructions for setting up StoryAudit on Windows, Linux, or macOS.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Configuration](#configuration)
4. [Verification](#verification)
5. [Advanced Options](#advanced-options)
6. [Troubleshooting](#troubleshooting)
7. [Performance Tuning](#performance-tuning)

## System Requirements

### Minimum
- **OS**: Windows 10+, Ubuntu 18+, macOS 10.14+
- **Python**: 3.9+ (check with `python --version`)
- **RAM**: 4GB
- **Disk**: 500MB free space (2GB with cache)
- **Internet**: Required for Gemini API

### Recommended
- **RAM**: 8GB+
- **Disk**: SSD with 2GB+ free space
- **Python**: 3.10 or 3.11
- **CPU**: Multi-core processor

### API Requirements
- Google Gemini API key (free at [ai.google.dev](https://ai.google.dev/))

## Installation Steps

### Step 1: Clone the Repository

```bash
# Using git
git clone https://github.com/yourrepo/StoryAudit.git
cd StoryAudit

# Or download ZIP and extract
# Then navigate to the folder
cd StoryAudit
```

### Step 2: Create Virtual Environment

A virtual environment isolates dependencies and prevents conflicts.

#### Windows
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

#### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Verify activation**: You should see `(.venv)` at the start of your terminal prompt.

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `sentence-transformers` - Embeddings
- `google-generativeai` - Gemini API
- `pandas` - Data processing
- `pathway` - Streaming pipeline
- And others (see `requirements.txt`)

**Installation time**: 3-5 minutes depending on internet speed.

### Step 4: Obtain Gemini API Key

1. Visit [ai.google.dev](https://ai.google.dev/)
2. Click "Get API Key" button
3. Select or create a Google Cloud project
4. Generate API key
5. Copy the key (you'll need it next)

### Step 5: Set Environment Variable

Store your API key securely so the system can access it.

#### Windows (PowerShell)
```powershell
# Set for current session
$env:GEMINI_API_KEY = "your-api-key-here"

# Set permanently (requires admin)
[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "your-api-key-here", "User")

# Verify it's set
$env:GEMINI_API_KEY
```

#### Windows (Command Prompt)
```cmd
set GEMINI_API_KEY=your-api-key-here
echo %GEMINI_API_KEY%
```

#### Linux / macOS (bash/zsh)
```bash
# Set for current session
export GEMINI_API_KEY="your-api-key-here"

# Set permanently (add to ~/.bashrc, ~/.zshrc, or ~/.bash_profile)
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc

# Verify it's set
echo $GEMINI_API_KEY
```

### Step 6: Verify Installation

Run the verification script:

```bash
python quick_test.py
```

Expected output:
```
Testing cache_manager...
✓ CacheManager initialized successfully
✓ Temporary cache entry added

Testing embeddings...
✓ Embeddings generated successfully
✓ Embedding shape: (384,)

Testing CSV loading...
✓ Test data loaded successfully
✓ 60 examples found

All tests passed! System is ready.
```

If you see errors, see the [Troubleshooting](#troubleshooting) section.

## Configuration

### Basic Configuration (Already Set)

Default settings in `config.py` are optimized for most use cases:

```python
# API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Processing
CHUNK_SIZE = 2500          # Words per chunk
CHUNK_OVERLAP = 300        # Word overlap between chunks
MAX_CLAIMS_PER_BACKSTORY = 20

# Cache
CACHE_DIR = ".storyaudit_cache"
ENABLE_CACHE = True
CACHE_TTL = 30 * 24 * 3600  # 30 days

# Paths (auto-detected)
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
```

### Optional Environment Variables

```bash
# Logging level (DEBUG, INFO, WARNING, ERROR)
export STORYAUDIT_LOG_LEVEL="INFO"

# Custom cache directory
export STORYAUDIT_CACHE_DIR="/path/to/cache"

# Disable cache (not recommended)
export STORYAUDIT_ENABLE_CACHE="false"

# Batch size for embeddings
export STORYAUDIT_BATCH_SIZE="16"
```

### Data Directory Structure

Ensure your data is organized correctly:

```
StoryAudit/
└── data/
    ├── The Count of Monte Cristo.txt
    └── In Search of the Castaways.txt
```

The novels should be plain text files with UTF-8 encoding.

## Verification

### Quick Verification (10 seconds)

```bash
python quick_test.py
```

### Full System Test (2 minutes)

```bash
# Process a single test example
python run.py --test-csv test.csv --output test_results.csv
```

Check results:
```bash
head test_results.csv
```

### Validation Command

```bash
# Validate environment
python run.py --validate
```

This checks:
- ✓ Python version
- ✓ Virtual environment
- ✓ Dependencies installed
- ✓ API key set
- ✓ Data files present
- ✓ Paths accessible

## Advanced Options

### Running with Optimizations

For faster processing with caching:

```bash
# First run (builds cache, ~45 minutes)
python run.py --test-csv test.csv --optimized

# Subsequent runs (uses cache, ~1-2 minutes)
python run.py --test-csv test.csv --optimized
```

### Clear Cache When Needed

```bash
# Delete all cached data
python run.py --test-csv test.csv --optimized --clear-cache

# Or manually
rm -rf .storyaudit_cache/
```

### Advanced Processing

```bash
# Verbose logging (show all processing steps)
python run.py --test-csv test.csv --optimized --verbose

# Advanced Track-A features
python run.py --test-csv test.csv --advanced --optimized

# Custom output file
python run.py --test-csv test.csv --output custom_results.csv --optimized
```

### Using Different CSV Files

```bash
# Process training data
python run.py --test-csv train.csv --optimized

# Custom CSV file
python run.py --test-csv path/to/data.csv --optimized

# Output to specific location
python run.py --test-csv data.csv --output results/output.csv
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'sentence_transformers'"

**Cause**: Dependencies not installed

**Solution**:
```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # Linux/macOS
.\.venv\Scripts\activate   # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "GEMINI_API_KEY not set"

**Cause**: API key not in environment

**Solution**:
```bash
# Verify it's set
echo $GEMINI_API_KEY  # Linux/macOS
echo %GEMINI_API_KEY% # Windows

# If empty, set it:
export GEMINI_API_KEY="your-key"  # Linux/macOS
set GEMINI_API_KEY=your-key       # Windows

# Verify again
echo $GEMINI_API_KEY
```

### Issue: "FileNotFoundError: [Errno 2] No such file or directory: 'data/...'"

**Cause**: Missing data files

**Solution**:
1. Ensure novels are in `data/` directory:
   ```bash
   ls data/  # Linux/macOS
   dir data\ # Windows
   ```

2. Check filenames match exactly (case-sensitive on Linux/macOS)

3. Verify file encoding is UTF-8:
   ```bash
   file data/*.txt  # Linux/macOS
   ```

### Issue: "MemoryError" or "Out of memory"

**Cause**: System doesn't have enough RAM

**Solutions**:
1. Use optimized mode (reuses cached data):
   ```bash
   python run.py --test-csv test.csv --optimized
   ```

2. Process smaller batches:
   ```bash
   # Only process first 10 examples
   head -n 11 test.csv > small_test.csv
   python run.py --test-csv small_test.csv --optimized
   ```

3. Close other applications
4. Increase virtual memory (advanced)

### Issue: "Slow processing on HDD"

**Cause**: Disk I/O bottleneck

**Solutions**:
1. Use optimized mode (minimizes disk reads):
   ```bash
   python run.py --test-csv test.csv --optimized
   ```

2. Move cache to SSD (if available):
   ```bash
   export STORYAUDIT_CACHE_DIR="/path/to/ssd/cache"
   ```

3. Check available disk space:
   ```bash
   df -h /  # Linux/macOS
   dir C:\  # Windows
   ```

### Issue: "Connection timeout" or API errors

**Cause**: Network issues or API overload

**Solutions**:
1. Check internet connection:
   ```bash
   ping google.com
   ```

2. Verify API key is valid:
   - Visit ai.google.dev
   - Check key hasn't expired
   - Regenerate if needed

3. Try again later (API may be temporarily overloaded)

4. Check your Google Cloud project quota

### Issue: "Python not found" or "command not found"

**Cause**: Python not in PATH

**Solutions**:

#### Windows
```bash
# Check Python installation
python --version

# If not found, reinstall Python
# During installation, check "Add Python to PATH"
```

#### Linux/macOS
```bash
# Use python3 instead
python3 --version
python3 -m pip install -r requirements.txt
python3 run.py --test-csv test.csv
```

### Issue: "Can't activate virtual environment"

**Cause**: Wrong activation command

**Solution**:
```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.\.venv\Scripts\activate.bat

# Linux/macOS
source .venv/bin/activate
```

## Performance Tuning

### Optimal Settings by Hardware

#### Low-End (4GB RAM, HDD)
```bash
# Use optimized mode, smaller batches
python run.py --test-csv test.csv --optimized

# Monitor cache size
du -sh .storyaudit_cache/

# Clear cache monthly
python run.py --test-csv test.csv --optimized --clear-cache
```

#### Mid-Range (8GB RAM, SSD)
```bash
# Default optimized mode is ideal
python run.py --test-csv test.csv --optimized
```

#### High-End (16GB+ RAM, Fast SSD)
```bash
# Can use additional features
python run.py --test-csv test.csv --advanced --optimized

# Process larger batches
```

### Monitoring Performance

```bash
# Check cache size
du -sh .storyaudit_cache/      # Linux/macOS
dir /s .storyaudit_cache\      # Windows

# Monitor during processing (second terminal)
# Linux/macOS:
watch -n 1 "du -sh .storyaudit_cache/"

# Windows PowerShell (second terminal):
while($true) { Clear-Host; (Get-ChildItem .storyaudit_cache -Recurse | Measure-Object -Sum Length).Sum / 1MB | Write-Host "Cache size (MB):"; Start-Sleep 5 }
```

### Cache Management

```bash
# View cache contents
ls .storyaudit_cache/          # Linux/macOS
dir .storyaudit_cache\         # Windows

# Clear specific cache component
rm -rf .storyaudit_cache/embeddings/    # Clear embeddings

# Clear all cache
rm -rf .storyaudit_cache/               # Clear everything
python run.py --test-csv test.csv --optimized --clear-cache
```

## Next Steps

1. Run the verification: `python quick_test.py`
2. Process test data: `python run.py --test-csv test.csv --optimized`
3. Check results: `head results.csv`
4. See [README.md](README.md) for full documentation
5. Review [requirements.txt](requirements.txt) for dependency details

## Getting Help

- **See errors in output?** Check the Troubleshooting section above
- **Need more info?** See [README.md](README.md) for full documentation
- **Performance questions?** See Performance Tuning section above
- **API issues?** Check [ai.google.dev](https://ai.google.dev/) documentation

## Quick Reference

| Task | Command |
|------|---------|
| Activate environment (Windows) | `.\.venv\Scripts\activate` |
| Activate environment (Linux/macOS) | `source .venv/bin/activate` |
| Set API key (Linux/macOS) | `export GEMINI_API_KEY="key"` |
| Set API key (Windows) | `set GEMINI_API_KEY=key` |
| Verify setup | `python quick_test.py` |
| Process data | `python run.py --test-csv test.csv --optimized` |
| Clear cache | `python run.py --test-csv test.csv --optimized --clear-cache` |
| Check results | `head results.csv` |

---

**Version**: 2.0 (Optimized)  
**Last Updated**: January 2026  
**Status**: Complete & Production Ready
