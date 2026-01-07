#!/usr/bin/env python3
"""
menu.py
Interactive menu-driven interface for StoryAudit
"""

import os
import sys
from pathlib import Path
from config import Config

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print application header."""
    print("\n" + "="*70)
    print(" "*15 + "STORYAUDIT - BACKSTORY CONSISTENCY CHECKER")
    print(" "*10 + "KDSH 2026 Track A: AI-Powered Narrative Analysis")
    print("="*70 + "\n")

def list_stories():
    """List all available stories."""
    narratives_dir = Config.NARRATIVES_DIR
    stories = {}
    
    if not narratives_dir.exists():
        print("❌ Narratives directory not found!")
        return {}
    
    for file in sorted(narratives_dir.glob("*.txt")):
        story_id = file.stem.split("_")[-1]
        try:
            story_id_num = int(story_id)
            size = file.stat().st_size / 1024  # KB
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                word_count = len(f.read().split())
            stories[story_id_num] = {"file": file.name, "size": size, "words": word_count}
        except:
            pass
    
    return stories

def print_stories(stories):
    """Print formatted story list."""
    if not stories:
        print("❌ No stories found in data/narratives/")
        return
    
    print("\n📚 AVAILABLE STORIES:\n")
    print(f"{'ID':<5} {'Filename':<25} {'Size':<10} {'Words':<8}")
    print("-" * 55)
    
    for story_id in sorted(stories.keys()):
        story = stories[story_id]
        print(f"{story_id:<5} {story['file']:<25} {story['size']:>6.1f}KB {story['words']:>7}")
    
    print()

def get_api_status():
    """Check if Google API key is set."""
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if api_key:
        masked = api_key[:10] + "*" * (len(api_key) - 14) + api_key[-4:]
        return True, masked
    return False, None

def print_status():
    """Print system status."""
    api_set, api_masked = get_api_status()
    
    print("\n🔧 SYSTEM STATUS:\n")
    
    # Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"✓ Python {py_version}")
    
    # API Key
    if api_set:
        print(f"✓ Google API Key: {api_masked}")
    else:
        print(f"❌ Google API Key: NOT SET")
        print(f"   Run: $env:GOOGLE_API_KEY=\"your-key-here\"")
    
    # Data directories
    if Config.NARRATIVES_DIR.exists():
        count = len(list(Config.NARRATIVES_DIR.glob("*.txt")))
        print(f"✓ Narratives: {count} stories")
    else:
        print(f"❌ Narratives directory: NOT FOUND")
    
    if Config.BACKSTORIES_DIR.exists():
        count = len(list(Config.BACKSTORIES_DIR.glob("*.txt")))
        print(f"✓ Backstories: {count} stories")
    else:
        print(f"❌ Backstories directory: NOT FOUND")
    
    print()

def menu_analyze():
    """Menu: Analyze a story."""
    stories = list_stories()
    print_stories(stories)
    
    if not stories:
        print("No stories available to analyze.")
        return
    
    api_set, _ = get_api_status()
    if not api_set:
        print("⚠️  Google API Key not set. Please set it first:")
        print("   $env:GOOGLE_API_KEY=\"your-key-here\"")
        return
    
    try:
        choice = input("Enter story ID to analyze (or 'all' for all stories, 'back' to return): ").strip().lower()
        
        if choice == "back":
            return
        
        if choice == "all":
            import subprocess
            result = subprocess.run([sys.executable, "run.py", "--all", "--verbose"], cwd=Config.PROJECT_ROOT)
            sys.exit(result.returncode)
        
        if choice.isdigit():
            story_id = int(choice)
            if story_id not in stories:
                print(f"❌ Story ID {story_id} not found!")
                return
            
            import subprocess
            result = subprocess.run([sys.executable, "run.py", "--story-id", str(story_id), "--verbose"], 
                                  cwd=Config.PROJECT_ROOT)
            sys.exit(result.returncode)
        else:
            print("❌ Invalid choice!")
    
    except KeyboardInterrupt:
        print("\n⏸️  Cancelled.")

def menu_add_story():
    """Menu: Add a new story."""
    print("\n📝 ADD NEW STORY\n")
    
    try:
        story_id = input("Enter story ID (numeric, e.g., 5): ").strip()
        
        if not story_id.isdigit():
            print("❌ Story ID must be numeric!")
            return
        
        story_id = int(story_id)
        
        # Check if already exists
        narrative_file = Config.NARRATIVES_DIR / f"story_{story_id}.txt"
        backstory_file = Config.BACKSTORIES_DIR / f"backstory_{story_id}.txt"
        
        if narrative_file.exists():
            print(f"⚠️  Story {story_id} already exists!")
            overwrite = input("Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                return
        
        print("\n📖 Enter the NARRATIVE (novel/story text):")
        print("(Type 'END' on a new line when finished)\n")
        
        narrative_lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            narrative_lines.append(line)
        
        if not narrative_lines:
            print("❌ Narrative cannot be empty!")
            return
        
        print("\n📋 Enter the BACKSTORY (character background):")
        print("(Type 'END' on a new line when finished)\n")
        
        backstory_lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            backstory_lines.append(line)
        
        if not backstory_lines:
            print("❌ Backstory cannot be empty!")
            return
        
        # Save files
        Config.NARRATIVES_DIR.mkdir(parents=True, exist_ok=True)
        Config.BACKSTORIES_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(narrative_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(narrative_lines))
        
        with open(backstory_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(backstory_lines))
        
        print(f"\n✅ Story {story_id} saved successfully!")
        print(f"   Narrative: {len(narrative_lines)} lines")
        print(f"   Backstory: {len(backstory_lines)} lines")
    
    except KeyboardInterrupt:
        print("\n⏸️  Cancelled.")

def menu_view_results():
    """Menu: View analysis results."""
    results_file = Config.RESULTS_FILE
    
    if not results_file.exists():
        print("\n❌ No results file found yet. Analyze a story first.")
        return
    
    print("\n📊 ANALYSIS RESULTS:\n")
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            content = f.read()
        print(content)
    except Exception as e:
        print(f"❌ Error reading results: {e}")

def menu_set_api_key():
    """Menu: Set API key."""
    print("\n🔑 SET GOOGLE API KEY\n")
    
    print("Get a free API key from: https://ai.google.dev/")
    print("(No credit card required for free tier)\n")
    
    try:
        api_key = input("Enter your Google API key (or press Enter to skip): ").strip()
        
        if not api_key:
            print("Skipped.")
            return
        
        # Set it for current session
        os.environ["GOOGLE_API_KEY"] = api_key
        
        # Also save to .env file
        env_file = Config.PROJECT_ROOT / ".env"
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(f'GOOGLE_API_KEY="{api_key}"\n')
        
        masked = api_key[:10] + "*" * (len(api_key) - 14) + api_key[-4:]
        print(f"\n✅ API key set: {masked}")
        print("✅ Saved to .env file")
    
    except KeyboardInterrupt:
        print("\n⏸️  Cancelled.")

def main():
    """Main menu loop."""
    clear_screen()
    
    while True:
        clear_screen()
        print_header()
        print_status()
        
        print("MENU OPTIONS:\n")
        print("1️⃣  Analyze Stories (Check consistency)")
        print("2️⃣  Add New Story (Manual entry)")
        print("3️⃣  View Results (Last analysis)")
        print("4️⃣  Set API Key")
        print("5️⃣  System Status")
        print("6️⃣  Exit\n")
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == "1":
            menu_analyze()
        elif choice == "2":
            menu_add_story()
        elif choice == "3":
            menu_view_results()
        elif choice == "4":
            menu_set_api_key()
        elif choice == "5":
            print_status()
            input("Press Enter to continue...")
        elif choice == "6":
            print("\n👋 Goodbye!\n")
            break
        else:
            print("❌ Invalid choice!")
            input("Press Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Application terminated.\n")
        sys.exit(0)
