"""
Tokenizer training script for LLM training.
Trains a SentencePiece tokenizer on cleaned text data.
"""

import os
import sys
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

# Tokenizer imports
try:
    import sentencepiece as spm
except ImportError:
    logger.error("SentencePiece not available. Please install with: pip install sentencepiece")
    sys.exit(1)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from training.utils import setup_logging, ConfigManager


class TokenizerTrainer:
    """Handles tokenizer training using SentencePiece."""
    
    def __init__(self, 
                 vocab_size: int = 16000,
                 model_type: str = 'bpe',
                 character_coverage: float = 0.9995,
                 pad_id: int = 0,
                 unk_id: int = 1,
                 bos_id: int = 2,
                 eos_id: int = 3):
        """
        Initialize tokenizer trainer.
        
        Args:
            vocab_size: Size of vocabulary
            model_type: Type of tokenizer ('bpe', 'unigram', 'char', 'word')
            character_coverage: Character coverage for training
            pad_id: Padding token ID
            unk_id: Unknown token ID
            bos_id: Beginning of sequence token ID
            eos_id: End of sequence token ID
        """
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        
        logger.info(f"Tokenizer trainer initialized")
        logger.info(f"Vocab size: {vocab_size}, Model type: {model_type}")
    
    def prepare_training_data(self, data_dir: str = "data/cleaned") -> Path:
        """Prepare training data for tokenizer."""
        data_dir = Path(data_dir)
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Cleaned data directory not found: {data_dir}")
        
        # Look for combined text file first
        combined_file = data_dir / "combined_text.txt"
        if combined_file.exists():
            logger.info(f"Using combined text file: {combined_file}")
            return combined_file
        
        # Otherwise, combine all text files
        text_files = list(data_dir.glob("*.txt"))
        
        if not text_files:
            raise FileNotFoundError(f"No text files found in {data_dir}")
        
        # Create combined file
        logger.info(f"Combining {len(text_files)} text files...")
        
        with open(combined_file, 'w', encoding='utf-8') as outfile:
            for text_file in text_files:
                try:
                    with open(text_file, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                        outfile.write('\n\n')
                    logger.info(f"Added {text_file.name}")
                except Exception as e:
                    logger.error(f"Error reading {text_file}: {e}")
                    continue
        
        logger.info(f"Combined training data saved to: {combined_file}")
        return combined_file
    
    def train_tokenizer(self, 
                       input_file: Path, 
                       output_dir: str = "tokenizer",
                       model_prefix: str = "tokenizer") -> Dict[str, Any]:
        """Train SentencePiece tokenizer."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / model_prefix
        
        # SentencePiece training arguments
        train_args = [
            f'--input={input_file}',
            f'--model_prefix={model_path}',
            f'--vocab_size={self.vocab_size}',
            f'--model_type={self.model_type}',
            f'--character_coverage={self.character_coverage}',
            f'--pad_id={self.pad_id}',
            f'--unk_id={self.unk_id}',
            f'--bos_id={self.bos_id}',
            f'--eos_id={self.eos_id}',
            '--add_dummy_prefix=true',
            '--remove_extra_whitespaces=true',
            '--normalization_rule_name=identity'
        ]
        
        logger.info("Starting tokenizer training...")
        logger.info(f"Training arguments: {' '.join(train_args)}")
        
        try:
            # Train tokenizer
            spm.SentencePieceTrainer.train(' '.join(train_args))
            
            # Verify output files
            model_file = Path(f"{model_path}.model")
            vocab_file = Path(f"{model_path}.vocab")
            
            if not model_file.exists() or not vocab_file.exists():
                raise FileNotFoundError("Tokenizer training failed - output files not found")
            
            logger.info(f"Tokenizer training completed successfully!")
            logger.info(f"Model file: {model_file}")
            logger.info(f"Vocab file: {vocab_file}")
            
            # Test the tokenizer
            test_results = self.test_tokenizer(model_file)
            
            return {
                'model_file': model_file,
                'vocab_file': vocab_file,
                'vocab_size': self.vocab_size,
                'model_type': self.model_type,
                **test_results
            }
            
        except Exception as e:
            logger.error(f"Tokenizer training failed: {e}")
            raise
    
    def test_tokenizer(self, model_file: Path) -> Dict[str, Any]:
        """Test the trained tokenizer."""
        logger.info("Testing tokenizer...")
        
        try:
            # Load tokenizer
            sp = spm.SentencePieceProcessor()
            sp.load(str(model_file))
            
            # Test sentences
            test_sentences = [
                "Hello, world!",
                "This is a test sentence for the tokenizer.",
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning and artificial intelligence are fascinating."
            ]
            
            results = []
            total_tokens = 0
            
            for sentence in test_sentences:
                # Encode
                tokens = sp.encode(sentence, out_type=int)
                token_pieces = sp.encode(sentence, out_type=str)
                
                # Decode
                decoded = sp.decode(tokens)
                
                result = {
                    'original': sentence,
                    'tokens': tokens,
                    'token_pieces': token_pieces,
                    'decoded': decoded,
                    'num_tokens': len(tokens)
                }
                
                results.append(result)
                total_tokens += len(tokens)
                
                logger.info(f"'{sentence}' -> {len(tokens)} tokens")
                logger.info(f"Tokens: {token_pieces}")
            
            # Calculate average tokens per sentence
            avg_tokens = total_tokens / len(test_sentences)
            
            logger.info(f"Tokenizer test completed. Average tokens per sentence: {avg_tokens:.1f}")
            
            return {
                'test_results': results,
                'avg_tokens_per_sentence': avg_tokens,
                'actual_vocab_size': sp.get_piece_size()
            }
            
        except Exception as e:
            logger.error(f"Tokenizer test failed: {e}")
            return {'test_error': str(e)}
    
    def create_tokenized_dataset(self, 
                                input_file: Path,
                                model_file: Path,
                                output_dir: str = "data/tokens") -> Path:
        """Create tokenized dataset for training."""
        logger.info("Creating tokenized dataset...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load tokenizer
            sp = spm.SentencePieceProcessor()
            sp.load(str(model_file))
            
            # Read input text
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            logger.info(f"Input text: {len(text):,} characters")
            
            # Tokenize text
            tokens = sp.encode(text, out_type=int)
            
            logger.info(f"Tokenized: {len(tokens):,} tokens")
            
            # Convert to tensor and save
            token_tensor = torch.tensor(tokens, dtype=torch.long)
            output_file = output_dir / "tokens.pt"
            
            torch.save(token_tensor, output_file)
            
            logger.info(f"Tokenized dataset saved to: {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Dataset tokenization failed: {e}")
            raise


def main():
    """Main entry point."""
    # Setup logging
    setup_logging(log_dir="logs", level="INFO")
    
    try:
        # Load configuration
        config = ConfigManager.load_config("training/config.yaml")
        vocab_size = config['model']['vocab_size']
        
        # Initialize trainer
        trainer = TokenizerTrainer(
            vocab_size=vocab_size,
            model_type='bpe',  # Byte-pair encoding
            character_coverage=0.9995
        )
        
        # Prepare training data
        logger.info("Preparing training data...")
        input_file = trainer.prepare_training_data()
        
        # Train tokenizer
        logger.info("Training tokenizer...")
        results = trainer.train_tokenizer(input_file)
        
        # Create tokenized dataset
        logger.info("Creating tokenized dataset...")
        dataset_file = trainer.create_tokenized_dataset(
            input_file, results['model_file']
        )
        
        # Print summary
        logger.info("=== Tokenizer Training Summary ===")
        logger.info(f"Vocabulary size: {results['vocab_size']}")
        logger.info(f"Model type: {results['model_type']}")
        logger.info(f"Model file: {results['model_file']}")
        logger.info(f"Vocab file: {results['vocab_file']}")
        logger.info(f"Dataset file: {dataset_file}")
        
        if 'actual_vocab_size' in results:
            logger.info(f"Actual vocab size: {results['actual_vocab_size']}")
        
        if 'avg_tokens_per_sentence' in results:
            logger.info(f"Avg tokens per sentence: {results['avg_tokens_per_sentence']:.1f}")
        
        logger.info("Tokenizer training completed successfully!")
        
    except Exception as e:
        logger.error(f"Tokenizer training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

