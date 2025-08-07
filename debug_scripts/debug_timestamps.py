#!/usr/bin/env python3
"""
Quick debug script to check file timestamps
"""
from pathlib import Path

raw_dir = Path("data/raw")
cleaned_dir = Path("data/cleaned")

print("Checking first few files:")

# Get files with supported extensions
supported_extensions = ['.txt', '.pdf', '.docx', '.md', '.pptx']
raw_files = []
for ext in supported_extensions:
    raw_files.extend(raw_dir.glob(f"**/*{ext}"))

raw_files = raw_files[:5]  # First 5 files

for raw_file in raw_files:
    cleaned_file = cleaned_dir / f"{raw_file.stem}_cleaned.txt"
    
    print(f"\nRaw: {raw_file.name}")
    print(f"Cleaned: {cleaned_file.name}")
    print(f"Cleaned exists: {cleaned_file.exists()}")
    
    if cleaned_file.exists():
        raw_mtime = raw_file.stat().st_mtime
        cleaned_mtime = cleaned_file.stat().st_mtime
        
        print(f"Raw mtime: {raw_mtime}")
        print(f"Cleaned mtime: {cleaned_mtime}")
        print(f"Cleaned newer: {cleaned_mtime >= raw_mtime}")
        print(f"Should skip: {cleaned_mtime >= raw_mtime}")
    else:
        print("Should process: No cleaned file")