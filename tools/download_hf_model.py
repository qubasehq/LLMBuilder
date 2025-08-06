#!/usr/bin/env python3
"""
Download all files for a HuggingFace model repository (main branch) to a local directory.
Usage:
    python tools/download_hf_model.py --model Qwen/Qwen2.5-Coder-0.5B --output-dir ./models/Qwen2.5-Coder-0.5B
"""
import os
import sys
import argparse
import requests
import json
from pathlib import Path


def fetch_model_file_list(model_name: str):
    api_url = f"https://huggingface.co/api/models/{model_name}/tree/main"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] Failed to fetch file list: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)


def download_file(url: str, save_path: Path):
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"[OK] {save_path}")
    except Exception as e:
        print(f"[FAIL] {save_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download all files for a HuggingFace model repository.")
    parser.add_argument("--model", type=str, required=True, help="Model repo name, e.g. Qwen/Qwen2.5-Coder-0.5B")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save files (default: ./models/<model_name>)")
    args = parser.parse_args()

    model_name = args.model.strip()
    output_dir = args.output_dir or os.path.join("models", model_name.replace('/', '_'))
    output_dir = os.path.abspath(output_dir)
    base_url = f"https://huggingface.co/{model_name}/resolve/main"

    print(f"Fetching file list for: {model_name}")
    files_data = fetch_model_file_list(model_name)
    all_files = [item['path'] for item in files_data if item['type'] == 'file']
    print(f"Found {len(all_files)} files.")

    # Save file list for reference
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, "available_files.json"), "w") as f:
        json.dump(all_files, f, indent=2)

    print(f"\nDownloading files to: {output_dir}\n{'='*40}")
    for filename in all_files:
        url = f"{base_url}/{filename}"
        save_path = Path(output_dir) / filename
        print(f"Downloading: {filename}")
        download_file(url, save_path)

    print(f"\n[Done] All files downloaded to: {output_dir}")


if __name__ == "__main__":
    main()
