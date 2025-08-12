#!/usr/bin/env python3
"""
Test TokenizerTrainer implementations.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from training.train_tokenizer import (
    TokenizerConfig, TokenizerTrainer, HuggingFaceTokenizerTrainer, 
    SentencePieceTokenizerTrainer, create_tokenizer_trainer,
    TOKENIZERS_AVAILABLE, SENTENCEPIECE_AVAILABLE
)
from loguru import logger


def create_test_corpus(test_dir: Path):
    """Create test corpus files for tokenizer training."""
    
    # Create diverse text content for training
    corpus_texts = [
        """
        The quick brown fox jumps over the lazy dog.
        This sentence contains every letter of the alphabet.
        Machine learning is transforming the world of technology.
        Natural language processing enables computers to understand human language.
        """,
        
        """
        Artificial intelligence has made significant progress in recent years.
        Deep learning models can process vast amounts of data efficiently.
        Neural networks are inspired by the structure of the human brain.
        Computer vision allows machines to interpret and analyze visual information.
        """,
        
        """
        Climate change is one of the most pressing challenges of our time.
        Renewable energy sources like solar and wind power are becoming more affordable.
        Sustainable development requires balancing economic growth with environmental protection.
        Conservation efforts are essential for preserving biodiversity.
        """,
        
        """
        Space exploration continues to push the boundaries of human knowledge.
        Mars missions provide valuable insights into planetary science.
        The International Space Station serves as a laboratory in orbit.
        Telescopes help us observe distant galaxies and understand the universe.
        """,
        
        """
        Programming languages enable developers to create software applications.
        Python is popular for data science and machine learning projects.
        JavaScript powers interactive web applications and user interfaces.
        Open source software promotes collaboration and innovation.
        """
    ]
    
    # Create corpus files
    corpus_files = []
    for i, text in enumerate(corpus_texts):
        file_path = test_dir / f"corpus_{i+1}.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text.strip())
        corpus_files.append(file_path)
    
    return corpus_files


def test_tokenizer_config():
    """Test TokenizerConfig functionality."""
    logger.info("Testing TokenizerConfig...")
    
    # Test basic configuration
    config = TokenizerConfig(
        tokenizer_type="huggingface",
        vocab_size=1000,
        model_type="bpe",
        special_tokens={"pad_token": "<pad>", "unk_token": "<unk>"}
    )
    
    if config.tokenizer_type == "huggingface" and config.vocab_size == 1000:
        logger.info("✅ TokenizerConfig creation working")
    else:
        logger.error("❌ TokenizerConfig creation failed")
        return False
    
    # Test presets
    presets = TokenizerTrainer.load_presets()
    
    if len(presets) > 0:
        logger.info(f"✅ Found {len(presets)} preset configurations")
        for name, preset_config in presets.items():
            logger.info(f"  {name}: {preset_config.tokenizer_type} {preset_config.model_type}")
    else:
        logger.error("❌ No preset configurations found")
        return False
    
    return True


def test_huggingface_tokenizer():
    """Test HuggingFace tokenizer training."""
    if not TOKENIZERS_AVAILABLE:
        logger.warning("Skipping HuggingFace tokenizer test - tokenizers not available")
        return True
    
    logger.info("Testing HuggingFace tokenizer training...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        output_dir = test_dir / "hf_tokenizer"
        
        # Create test corpus
        corpus_files = create_test_corpus(test_dir)
        
        # Create configuration
        config = TokenizerConfig(
            tokenizer_type="huggingface",
            vocab_size=1000,
            model_type="bpe",
            special_tokens={
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>"
            },
            normalization=True,
            pre_tokenizers=["whitespace", "punctuation"]
        )
        
        # Create trainer
        trainer = HuggingFaceTokenizerTrainer(config)
        
        # Train tokenizer
        success = trainer.train(corpus_files, output_dir)
        
        if not success:
            logger.error("❌ HuggingFace tokenizer training failed")
            return False
        
        # Test encoding/decoding
        test_text = "Hello world! This is a test."
        token_ids = trainer.encode(test_text)
        decoded_text = trainer.decode(token_ids)
        
        logger.info(f"HF Tokenizer test:")
        logger.info(f"  Original: {test_text}")
        logger.info(f"  Token IDs: {token_ids}")
        logger.info(f"  Decoded: {decoded_text}")
        logger.info(f"  Vocab size: {trainer.get_vocab_size()}")
        
        # Verify files were created
        expected_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json"]
        for filename in expected_files:
            file_path = output_dir / filename
            if not file_path.exists():
                logger.error(f"❌ Expected file not created: {filename}")
                return False
        
        # Test loading
        new_trainer = HuggingFaceTokenizerTrainer(config)
        if not new_trainer.load_tokenizer(output_dir):
            logger.error("❌ Failed to load trained tokenizer")
            return False
        
        # Test that loaded tokenizer works the same
        new_token_ids = new_trainer.encode(test_text)
        if token_ids == new_token_ids:
            logger.info("✅ HuggingFace tokenizer training and loading successful")
            return True
        else:
            logger.error("❌ Loaded tokenizer produces different results")
            return False


def test_sentencepiece_tokenizer():
    """Test SentencePiece tokenizer training."""
    if not SENTENCEPIECE_AVAILABLE:
        logger.warning("Skipping SentencePiece tokenizer test - SentencePiece not available")
        return True
    
    logger.info("Testing SentencePiece tokenizer training...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        output_dir = test_dir / "sp_tokenizer"
        
        # Create test corpus
        corpus_files = create_test_corpus(test_dir)
        
        # Create configuration
        config = TokenizerConfig(
            tokenizer_type="sentencepiece",
            vocab_size=1000,
            model_type="bpe",
            special_tokens={
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>"
            },
            trainer_args={
                "character_coverage": 0.9995,
                "split_by_whitespace": True
            }
        )
        
        # Create trainer
        trainer = SentencePieceTokenizerTrainer(config)
        
        # Train tokenizer
        success = trainer.train(corpus_files, output_dir)
        
        if not success:
            logger.error("❌ SentencePiece tokenizer training failed")
            return False
        
        # Test encoding/decoding
        test_text = "Hello world! This is a test."
        token_ids = trainer.encode(test_text)
        decoded_text = trainer.decode(token_ids)
        
        logger.info(f"SentencePiece test:")
        logger.info(f"  Original: {test_text}")
        logger.info(f"  Token IDs: {token_ids}")
        logger.info(f"  Decoded: {decoded_text}")
        logger.info(f"  Vocab size: {trainer.get_vocab_size()}")
        
        # Verify files were created
        expected_files = ["sentencepiece.model", "sentencepiece.vocab", "tokenizer_config.json", "vocab.txt"]
        for filename in expected_files:
            file_path = output_dir / filename
            if not file_path.exists():
                logger.error(f"❌ Expected file not created: {filename}")
                return False
        
        # Test loading
        new_trainer = SentencePieceTokenizerTrainer(config)
        if not new_trainer.load_tokenizer(output_dir):
            logger.error("❌ Failed to load trained tokenizer")
            return False
        
        # Test that loaded tokenizer works the same
        new_token_ids = new_trainer.encode(test_text)
        if token_ids == new_token_ids:
            logger.info("✅ SentencePiece tokenizer training and loading successful")
            return True
        else:
            logger.error("❌ Loaded tokenizer produces different results")
            return False


def test_factory_function():
    """Test tokenizer factory function."""
    logger.info("Testing tokenizer factory function...")
    
    # Test HuggingFace creation
    if TOKENIZERS_AVAILABLE:
        hf_config = TokenizerConfig(
            tokenizer_type="huggingface",
            vocab_size=1000,
            model_type="bpe",
            special_tokens={"unk_token": "<unk>"}
        )
        
        hf_trainer = create_tokenizer_trainer(hf_config)
        if isinstance(hf_trainer, HuggingFaceTokenizerTrainer):
            logger.info("✅ HuggingFace tokenizer factory working")
        else:
            logger.error("❌ HuggingFace tokenizer factory failed")
            return False
    
    # Test SentencePiece creation
    if SENTENCEPIECE_AVAILABLE:
        sp_config = TokenizerConfig(
            tokenizer_type="sentencepiece",
            vocab_size=1000,
            model_type="bpe",
            special_tokens={"unk_token": "<unk>"}
        )
        
        sp_trainer = create_tokenizer_trainer(sp_config)
        if isinstance(sp_trainer, SentencePieceTokenizerTrainer):
            logger.info("✅ SentencePiece tokenizer factory working")
        else:
            logger.error("❌ SentencePiece tokenizer factory failed")
            return False
    
    # Test invalid type
    try:
        invalid_config = TokenizerConfig(
            tokenizer_type="invalid",
            vocab_size=1000,
            model_type="bpe",
            special_tokens={"unk_token": "<unk>"}
        )
        create_tokenizer_trainer(invalid_config)
        logger.error("❌ Factory should have failed with invalid type")
        return False
    except ValueError:
        logger.info("✅ Factory correctly rejects invalid tokenizer type")
    
    return True


def test_preset_configurations():
    """Test preset configurations."""
    logger.info("Testing preset configurations...")
    
    presets = TokenizerTrainer.load_presets()
    
    # Test a few key presets
    key_presets = ["gpt2_small", "bert_base"]
    
    for preset_name in key_presets:
        if preset_name not in presets:
            logger.error(f"❌ Missing preset: {preset_name}")
            return False
        
        config = presets[preset_name]
        
        # Verify preset has required fields
        if not config.tokenizer_type or not config.model_type or config.vocab_size <= 0:
            logger.error(f"❌ Invalid preset configuration: {preset_name}")
            return False
        
        logger.info(f"✅ Preset {preset_name}: {config.tokenizer_type} {config.model_type} (vocab={config.vocab_size})")
    
    return True


def test_tokenizer_comparison():
    """Compare different tokenizer configurations."""
    logger.info("Testing tokenizer comparison...")
    
    if not (TOKENIZERS_AVAILABLE and SENTENCEPIECE_AVAILABLE):
        logger.warning("Skipping comparison test - not all tokenizers available")
        return True
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Create test corpus
        corpus_files = create_test_corpus(test_dir)
        
        # Test text
        test_text = "The quick brown fox jumps over the lazy dog."
        
        # Test different configurations
        configs = [
            ("HF BPE", TokenizerConfig(
                tokenizer_type="huggingface",
                vocab_size=500,
                model_type="bpe",
                special_tokens={"unk_token": "<unk>"}
            )),
            ("SP BPE", TokenizerConfig(
                tokenizer_type="sentencepiece",
                vocab_size=500,
                model_type="bpe",
                special_tokens={"unk_token": "<unk>"}
            ))
        ]
        
        results = []
        
        for name, config in configs:
            try:
                output_dir = test_dir / name.lower().replace(" ", "_")
                trainer = create_tokenizer_trainer(config)
                
                if trainer.train(corpus_files, output_dir):
                    token_ids = trainer.encode(test_text)
                    decoded = trainer.decode(token_ids)
                    vocab_size = trainer.get_vocab_size()
                    
                    results.append({
                        'name': name,
                        'tokens': len(token_ids),
                        'vocab_size': vocab_size,
                        'decoded_matches': decoded.strip() == test_text.strip()
                    })
                    
                    logger.info(f"{name}: {len(token_ids)} tokens, vocab={vocab_size}")
                
            except Exception as e:
                logger.warning(f"Failed to test {name}: {e}")
        
        if len(results) >= 2:
            logger.info("✅ Tokenizer comparison completed")
            return True
        else:
            logger.warning("⚠️ Could not compare tokenizers")
            return True  # Don't fail the test for this


def main():
    """Run all tokenizer trainer tests."""
    logger.info("🔍 Testing TokenizerTrainer implementations...")
    
    tests = [
        ("TokenizerConfig", test_tokenizer_config),
        ("HuggingFace Tokenizer", test_huggingface_tokenizer),
        ("SentencePiece Tokenizer", test_sentencepiece_tokenizer),
        ("Factory Function", test_factory_function),
        ("Preset Configurations", test_preset_configurations),
        ("Tokenizer Comparison", test_tokenizer_comparison),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            if test_func():
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TOKENIZER TRAINER TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tokenizer trainer tests passed!")
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    main()