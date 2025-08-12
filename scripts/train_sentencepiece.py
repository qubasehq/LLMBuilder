#!/usr/bin/env python3
"""
Dedicated CLI script for SentencePiece tokenizer training.
Provides enhanced configuration options and error handling.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from training.train_tokenizer import SentencePieceTokenizerTrainer, TokenizerConfig, SENTENCEPIECE_AVAILABLE
from loguru import logger


def estimate_vocab_size(input_files: list, target_vocab_size: int) -> int:
    """Estimate appropriate vocab size based on corpus."""
    total_chars = 0
    unique_words = set()
    
    for file_path in input_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                total_chars += len(content)
                words = content.lower().split()
                unique_words.update(words)
        except Exception as e:
            logger.warning(f"Could not read {file_path} for estimation: {e}")
    
    logger.info(f"Corpus statistics:")
    logger.info(f"  Total characters: {total_chars:,}")
    logger.info(f"  Unique words: {len(unique_words):,}")
    
    # Conservative estimate: vocab should be much smaller than unique words
    max_reasonable_vocab = min(len(unique_words) // 2, target_vocab_size)
    
    if total_chars < 10000:  # Very small corpus
        recommended = min(max_reasonable_vocab, 100)
    elif total_chars < 100000:  # Small corpus
        recommended = min(max_reasonable_vocab, 1000)
    else:
        recommended = target_vocab_size
    
    if recommended != target_vocab_size:
        logger.info(f"Recommended vocab_size: {recommended} (requested: {target_vocab_size})")
    
    return recommended


def create_sentencepiece_config(args) -> TokenizerConfig:
    """Create SentencePiece configuration from CLI arguments."""
    
    # Special tokens
    special_tokens = {
        "pad_token": args.pad_token,
        "unk_token": args.unk_token,
        "bos_token": args.bos_token,
        "eos_token": args.eos_token
    }
    
    # Trainer arguments
    trainer_args = {
        "character_coverage": args.character_coverage,
        "split_by_whitespace": args.split_by_whitespace,
        "max_sentence_length": args.max_sentence_length,
        "num_threads": args.num_threads,
        "input_sentence_size": args.input_sentence_size,
        "shuffle_input_sentence": args.shuffle_input_sentence,
        "hard_vocab_limit": args.hard_vocab_limit
    }
    
    # Add optional arguments
    if args.shrinking_factor:
        trainer_args["shrinking_factor"] = args.shrinking_factor
    if args.max_sentencepiece_length:
        trainer_args["max_sentencepiece_length"] = args.max_sentencepiece_length
    
    config = TokenizerConfig(
        tokenizer_type="sentencepiece",
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        special_tokens=special_tokens,
        normalization=args.normalization,
        trainer_args=trainer_args
    )
    
    return config


def main():
    """Main CLI entry point for SentencePiece tokenizer training."""
    parser = argparse.ArgumentParser(
        description="Train SentencePiece tokenizer with comprehensive options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic BPE training
  python scripts/train_sentencepiece.py --input-dir data/cleaned --output-dir tokenizer --vocab-size 16000
  
  # Unigram model with custom settings
  python scripts/train_sentencepiece.py --input-dir data/cleaned --output-dir tokenizer --model-type unigram --vocab-size 32000 --character-coverage 0.999
  
  # Training with custom special tokens
  python scripts/train_sentencepiece.py --input-dir data/cleaned --output-dir tokenizer --bos-token "<start>" --eos-token "<end>"
        """
    )
    
    # Required arguments
    parser.add_argument("--input-dir", required=True, help="Input directory with text files")
    parser.add_argument("--output-dir", required=True, help="Output directory for tokenizer")
    parser.add_argument("--vocab-size", type=int, default=16000, help="Vocabulary size (default: 16000)")
    
    # Model configuration
    parser.add_argument("--model-type", choices=["bpe", "unigram"], default="bpe", 
                       help="Model type (default: bpe)")
    parser.add_argument("--normalization", action="store_true", default=True,
                       help="Enable text normalization (default: True)")
    parser.add_argument("--no-normalization", action="store_false", dest="normalization",
                       help="Disable text normalization")
    
    # Special tokens
    parser.add_argument("--pad-token", default="<pad>", help="Padding token (default: <pad>)")
    parser.add_argument("--unk-token", default="<unk>", help="Unknown token (default: <unk>)")
    parser.add_argument("--bos-token", default="<s>", help="Beginning of sequence token (default: <s>)")
    parser.add_argument("--eos-token", default="</s>", help="End of sequence token (default: </s>)")
    
    # Training parameters
    parser.add_argument("--character-coverage", type=float, default=0.9995,
                       help="Character coverage (default: 0.9995)")
    parser.add_argument("--split-by-whitespace", action="store_true", default=True,
                       help="Split by whitespace (default: True)")
    parser.add_argument("--max-sentence-length", type=int, default=4192,
                       help="Maximum sentence length (default: 4192)")
    parser.add_argument("--num-threads", type=int, default=1,
                       help="Number of threads (default: 1)")
    parser.add_argument("--input-sentence-size", type=int, default=0,
                       help="Input sentence size limit (0=unlimited, default: 0)")
    parser.add_argument("--shuffle-input-sentence", action="store_true", default=True,
                       help="Shuffle input sentences (default: True)")
    parser.add_argument("--hard-vocab-limit", action="store_true", default=True,
                       help="Hard vocabulary limit (default: True)")
    
    # Advanced parameters
    parser.add_argument("--shrinking-factor", type=float,
                       help="Shrinking factor for vocabulary pruning")
    parser.add_argument("--max-sentencepiece-length", type=int,
                       help="Maximum sentence piece length")
    
    # Utility options
    parser.add_argument("--estimate-vocab", action="store_true",
                       help="Estimate appropriate vocab size and exit")
    parser.add_argument("--test-tokenizer", action="store_true",
                       help="Test the trained tokenizer with sample text")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Check if SentencePiece is available
    if not SENTENCEPIECE_AVAILABLE:
        logger.error("SentencePiece not available. Please install: pip install sentencepiece")
        sys.exit(1)
    
    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    # Find input files
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    input_files = list(input_dir.glob("*.txt"))
    if not input_files:
        logger.error(f"No .txt files found in {input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(input_files)} input files")
    
    # Estimate vocab size if requested
    if args.estimate_vocab:
        recommended = estimate_vocab_size(input_files, args.vocab_size)
        print(f"\nRecommended vocab_size: {recommended}")
        print(f"Requested vocab_size: {args.vocab_size}")
        if recommended != args.vocab_size:
            print(f"Consider using: --vocab-size {recommended}")
        return
    
    # Create configuration
    config = create_sentencepiece_config(args)
    
    # Create trainer
    try:
        trainer = SentencePieceTokenizerTrainer(config)
        output_dir = Path(args.output_dir)
        
        logger.info(f"Starting SentencePiece training...")
        logger.info(f"  Model type: {config.model_type}")
        logger.info(f"  Vocab size: {config.vocab_size}")
        logger.info(f"  Input files: {len(input_files)}")
        logger.info(f"  Output dir: {output_dir}")
        
        # Train tokenizer
        if trainer.train(input_files, output_dir):
            logger.info("✅ SentencePiece tokenizer training completed successfully!")
            
            # Show training statistics
            stats = trainer.get_training_stats()
            print(f"\n=== Training Results ===")
            print(f"Training time: {stats['training_time']:.2f}s")
            print(f"Vocabulary size: {stats['vocab_size']} (requested: {stats['requested_vocab_size']})")
            print(f"Corpus size: {stats['corpus_size_kb']:.1f} KB")
            print(f"Training speed: {stats['pieces_per_second']:.1f} pieces/second")
            
            # Test tokenizer if requested
            if args.test_tokenizer:
                test_texts = [
                    "Hello world! This is a test sentence.",
                    "Machine learning is transforming technology.",
                    "The quick brown fox jumps over the lazy dog."
                ]
                
                print(f"\n=== Tokenizer Test ===")
                for test_text in test_texts:
                    token_ids = trainer.encode(test_text)
                    decoded_text = trainer.decode(token_ids)
                    print(f"Original: {test_text}")
                    print(f"Tokens ({len(token_ids)}): {token_ids}")
                    print(f"Decoded: {decoded_text}")
                    print()
            
            print(f"Tokenizer saved to: {output_dir}")
            print(f"Files created:")
            for file_path in output_dir.glob("*"):
                print(f"  - {file_path.name}")
            
        else:
            logger.error("❌ SentencePiece tokenizer training failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()