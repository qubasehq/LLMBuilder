"""
Tokenize corpus text files into a single tensor of token IDs using SentencePiece.
Outputs a .pt file under data/tokens/ for training.
"""
import argparse
from pathlib import Path
from typing import List
import sys
import torch
from loguru import logger

# Ensure project root is importable
sys.path.append(str(Path(__file__).parent.parent))

try:
    import sentencepiece as spm
except ImportError as e:
    print("SentencePiece is required. pip install sentencepiece", file=sys.stderr)
    raise


def list_text_files(input_dir: Path) -> List[Path]:
    files = []
    for p in input_dir.rglob("*.txt"):
        if p.is_file() and p.stat().st_size > 0:
            files.append(p)
    return sorted(files)


def tokenize_files(sp: spm.SentencePieceProcessor, files: List[Path]) -> torch.Tensor:
    ids: List[int] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ids.extend(sp.encode_as_ids(line))
                    # Add EOS between lines to help boundaries
                    ids.append(sp.eos_id() if sp.eos_id() != -1 else 0)
        except Exception as e:
            logger.warning(f"Failed to read {fp}: {e}")
    if not ids:
        raise RuntimeError("No tokens produced from input files")
    return torch.tensor(ids, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Directory containing cleaned or deduped .txt files")
    ap.add_argument("--tokenizer", required=True, help="Path to SentencePiece .model file")
    ap.add_argument("--output", required=True, help="Path to output .pt file (e.g., data/tokens/tokens.pt)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    tok_path = Path(args.tokenizer)
    out_path = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {tok_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Listing text files in {input_dir}...")
    files = list_text_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")
    logger.info(f"Found {len(files)} text files")

    sp = spm.SentencePieceProcessor()
    sp.load(str(tok_path))
    logger.info(f"Loaded SentencePiece model: {tok_path}")

    logger.info("Tokenizing files...")
    tokens = tokenize_files(sp, files)
    logger.info(f"Tokenized {tokens.numel():,} tokens")

    torch.save(tokens, out_path)
    logger.info(f"Saved tokens to {out_path}")


if __name__ == "__main__":
    main()
