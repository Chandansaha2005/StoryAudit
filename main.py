#!/usr/bin/env python3
"""
main.py - Simple entry point for StoryAudit
Run this to start the interactive menu
"""

if __name__ == "__main__":
    import sys
    import subprocess
    
    # Try to run the menu
    try:
        from menu import main
        main()
    except KeyboardInterrupt:
        print("\n\nApplication terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTrying alternative startup method...")
        try:
            result = subprocess.run([sys.executable, "menu.py"])
            sys.exit(result.returncode)
        except Exception as e2:
            print(f"Failed to start: {e2}")
            sys.exit(1)
