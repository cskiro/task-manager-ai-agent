#!/usr/bin/env python3
"""Code formatting script using black and ruff."""

import subprocess
import sys
from pathlib import Path

def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🔧 {description}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    else:
        print(f"❌ {description} failed")
        if result.stderr:
            print(result.stderr)
        if result.stdout:
            print(result.stdout)
        return False

def main():
    """Run all formatting tools."""
    project_root = Path(__file__).parent.parent
    
    # Change to project root
    import os
    os.chdir(project_root)
    
    print("🚀 Running code formatters...\n")
    
    success = True
    
    # 1. Run ruff for import sorting and linting fixes
    success &= run_command(
        ["poetry", "run", "ruff", "check", "--fix", "src", "tests"],
        "Ruff import sorting and linting"
    )
    
    # 2. Run black for code formatting
    success &= run_command(
        ["poetry", "run", "black", "src", "tests"],
        "Black code formatting"
    )
    
    # 3. Run ruff again to check for any issues
    success &= run_command(
        ["poetry", "run", "ruff", "check", "src", "tests"],
        "Final ruff check"
    )
    
    # 4. Run mypy for type checking (informational only)
    print("\n📊 Running type checker (informational)...")
    subprocess.run(["poetry", "run", "mypy", "src"], capture_output=False)
    
    if success:
        print("\n✨ All formatting completed successfully!")
    else:
        print("\n⚠️  Some formatting steps failed. Please review the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()