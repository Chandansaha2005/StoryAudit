#!/usr/bin/env python
"""
verify_setup.py
Verify that the StoryAudit system is properly configured and ready to run.
This script checks:
1. All required packages are installed
2. Data files exist
3. Configuration is valid
4. Google API key is set
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version is 3.10 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"✗ Python 3.10+ required. You have {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor} (OK)")
    return True


def check_packages():
    """Check all required packages are installed."""
    required_packages = {
        'google.generativeai': 'Google Generative AI',
        'pathway': 'Pathway',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'dotenv': 'Python-dotenv'
    }
    
    all_ok = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} not installed")
            all_ok = False
    
    if not all_ok:
        print("\nInstall missing packages with:")
        print("  pip install google-generativeai pathway pandas numpy python-dotenv")
    
    return all_ok


def check_data_files():
    """Check that data files exist."""
    data_dir = Path(__file__).parent / "data"
    narratives_dir = data_dir / "narratives"
    backstories_dir = data_dir / "backstories"
    
    print("\nData Files:")
    
    # Check narratives
    if narratives_dir.exists():
        narrative_files = list(narratives_dir.glob("*.txt"))
        print(f"✓ Narratives directory: {len(narrative_files)} file(s)")
        for f in narrative_files[:3]:
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.1f} KB)")
    else:
        print("✗ Narratives directory not found")
        return False
    
    # Check backstories
    if backstories_dir.exists():
        backstory_files = list(backstories_dir.glob("*.txt"))
        print(f"✓ Backstories directory: {len(backstory_files)} file(s)")
        for f in backstory_files[:3]:
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.1f} KB)")
    else:
        print("✗ Backstories directory not found")
        return False
    
    if len(narrative_files) == 0 or len(backstory_files) == 0:
        print("✗ No data files found")
        return False
    
    return True


def check_api_key():
    """Check that Google API key is set."""
    api_key = os.getenv("GOOGLE_API_KEY", "")
    
    if not api_key:
        print("\n✗ GOOGLE_API_KEY environment variable not set")
        print("  Get a free API key from: https://ai.google.dev")
        print("  Then set it with:")
        print("    export GOOGLE_API_KEY=\"your-key-here\"  # Linux/macOS")
        print("    $env:GOOGLE_API_KEY=\"your-key-here\"    # PowerShell")
        return False
    
    # Mask the key in output
    masked = api_key[:6] + "*" * (len(api_key) - 10) + api_key[-4:]
    print(f"\n✓ Google API key is set ({masked})")
    return True


def check_config():
    """Check configuration file."""
    config_file = Path(__file__).parent / "config.py"
    
    if config_file.exists():
        content = config_file.read_text()
        if "GOOGLE_API_KEY" in content and "gemini-1.5-flash" in content:
            print("✓ Configuration file is correct")
            return True
        else:
            print("✗ Configuration file may have issues")
            return False
    else:
        print("✗ Configuration file not found")
        return False


def print_summary(results):
    """Print summary of checks."""
    print("\n" + "="*60)
    all_passed = all(results.values())
    
    if all_passed:
        print("✓ ALL CHECKS PASSED - Ready to run!")
        print("\nQuick start:")
        print("  python run.py --story-id 1")
        print("\nOther options:")
        print("  python run.py --all              # Process all stories")
        print("  python run.py --story-id 1 --verbose  # With details")
    else:
        print("✗ SETUP INCOMPLETE - Please fix issues above")
        print("\nFailing checks:")
        for check, passed in results.items():
            if not passed:
                print(f"  - {check}")
    
    print("="*60)
    return all_passed


def main():
    """Run all checks."""
    print("StoryAudit Setup Verification")
    print("="*60 + "\n")
    
    results = {}
    
    # Run all checks
    results['Python Version'] = check_python_version()
    
    print("\nPackages:")
    results['Required Packages'] = check_packages()
    
    results['Data Files'] = check_data_files()
    results['Google API Key'] = check_api_key()
    results['Configuration'] = check_config()
    
    # Print summary
    all_ok = print_summary(results)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
