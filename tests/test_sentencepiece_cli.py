#!/usr/bin/env python3
"""
Test SentencePiece CLI integration.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from training.train_tokenizer import SentencePieceTokenizerTrainer, TokenizerConfig, SENTENCEPIECE_AVAILABLE
from loguru import logger


def create_larger_test_corpus(test_dir: Path):
    """Create a larger test corpus to avoid vocab size issues."""
    
    # Create more substantial content
    corpus_texts = [
        """
        Natural language processing is a subfield of linguistics, computer science, and artificial intelligence 
        concerned with the interactions between computers and human language, in particular how to program 
        computers to process and analyze large amounts of natural language data. The goal is a computer 
        capable of understanding the contents of documents, including the contextual nuances of the language 
        within them. The technology can then accurately extract information and insights contained in the 
        documents as well as categorize and organize the documents themselves.
        """,
        
        """
        Machine learning is a method of data analysis that automates analytical model building. It is a branch 
        of artificial intelligence based on the idea that systems can learn from data, identify patterns and 
        make decisions with minimal human intervention. Machine learning algorithms build a model based on 
        training data in order to make predictions or decisions without being explicitly programmed to do so. 
        Machine learning algorithms are used in a wide variety of applications, such as in medicine, email 
        filtering, speech recognition, and computer vision.
        """,
        
        """
        Deep learning is part of a broader family of machine learning methods based on artificial neural 
        networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. 
        Deep learning architectures such as deep neural networks, deep belief networks, deep reinforcement 
        learning, recurrent neural networks, convolutional neural networks and transformers have been applied 
        to fields including computer vision, speech recognition, natural language processing, machine 
        translation, bioinformatics, drug design, medical image analysis, climate science, material inspection 
        and board game programs.
        """,
        
        """
        Artificial intelligence is intelligence demonstrated by machines, in contrast to the natural intelligence 
        displayed by humans and animals. Leading AI textbooks define the field as the study of intelligent agents: 
        any device that perceives its environment and takes actions that maximize its chance of successfully 
        achieving its goals. Colloquially, the term artificial intelligence is often used to describe machines 
        that mimic cognitive functions that humans associate with the human mind, such as learning and problem solving.
        """,
        
        """
        Computer vision is an interdisciplinary scientific field that deals with how computers can gain 
        high-level understanding from digital images or videos. From the perspective of engineering, it seeks 
        to understand and automate tasks that the human visual system can do. Computer vision tasks include 
        methods for acquiring, processing, analyzing and understanding digital images, and extraction of 
        high-dimensional data from the real world in order to produce numerical or symbolic information.
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


def test_sentencepiece_bpe():
    """Test SentencePiece BPE training."""
    if not SENTENCEPIECE_AVAILABLE:
        logger.warning("Skipping SentencePiece test - not available")
        return True
    
    logger.info("Testing SentencePiece BPE training...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        output_dir = test_dir / "sp_bpe"
        
        # Create larger test corpus
        corpus_files = create_larger_test_corpus(test_dir)
        
        # Create configuration with reasonable vocab size
        config = TokenizerConfig(
            tokenizer_type="sentencepiece",
            vocab_size=500,  # Reasonable size for test corpus
            model_type="bpe",
            special_tokens={
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>"
            },
            normalization=True,
            trainer_args={
                "character_coverage": 0.9995,
                "split_by_whitespace": True,
                "num_threads": 1,
                "hard_vocab_limit": True
            }
        )
        
        # Create trainer
        trainer = SentencePieceTokenizerTrainer(config)
        
        # Train tokenizer
        success = trainer.train(corpus_files, output_dir)
        
        if not success:
            logger.error("❌ SentencePiece BPE training failed")
            return False
        
        # Test encoding/decoding
        test_texts = [
            "Hello world!",
            "Machine learning is amazing.",
            "Natural language processing enables computers to understand text."
        ]
        
        for test_text in test_texts:
            token_ids = trainer.encode(test_text)
            decoded_text = trainer.decode(token_ids)
            
            logger.info(f"BPE Test:")
            logger.info(f"  Original: {test_text}")
            logger.info(f"  Tokens: {token_ids}")
            logger.info(f"  Decoded: {decoded_text}")
        
        # Verify artifacts were created
        expected_files = [
            "sentencepiece.model", "sentencepiece.vocab", 
            "tokenizer_config.json", "vocab.txt", "vocab_with_scores.txt", "model_info.json"
        ]
        
        for filename in expected_files:
            file_path = output_dir / filename
            if not file_path.exists():
                logger.error(f"❌ Expected file not created: {filename}")
                return False
        
        logger.info("✅ SentencePiece BPE training successful")
        return True


def test_sentencepiece_unigram():
    """Test SentencePiece Unigram training."""
    if not SENTENCEPIECE_AVAILABLE:
        logger.warning("Skipping SentencePiece Unigram test - not available")
        return True
    
    logger.info("Testing SentencePiece Unigram training...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        output_dir = test_dir / "sp_unigram"
        
        # Create test corpus
        corpus_files = create_larger_test_corpus(test_dir)
        
        # Create configuration for Unigram
        config = TokenizerConfig(
            tokenizer_type="sentencepiece",
            vocab_size=300,  # Smaller vocab for Unigram
            model_type="unigram",
            special_tokens={
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>"
            },
            normalization=True,
            trainer_args={
                "character_coverage": 0.999,
                "split_by_whitespace": True,
                "num_threads": 1
            }
        )
        
        # Create trainer
        trainer = SentencePieceTokenizerTrainer(config)
        
        # Train tokenizer
        success = trainer.train(corpus_files, output_dir)
        
        if not success:
            logger.error("❌ SentencePiece Unigram training failed")
            return False
        
        # Test encoding/decoding
        test_text = "Artificial intelligence is transforming technology."
        token_ids = trainer.encode(test_text)
        decoded_text = trainer.decode(token_ids)
        
        logger.info(f"Unigram Test:")
        logger.info(f"  Original: {test_text}")
        logger.info(f"  Tokens: {token_ids}")
        logger.info(f"  Decoded: {decoded_text}")
        logger.info(f"  Vocab size: {trainer.get_vocab_size()}")
        
        logger.info("✅ SentencePiece Unigram training successful")
        return True


def test_sentencepiece_loading():
    """Test loading trained SentencePiece models."""
    if not SENTENCEPIECE_AVAILABLE:
        logger.warning("Skipping SentencePiece loading test - not available")
        return True
    
    logger.info("Testing SentencePiece model loading...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        output_dir = test_dir / "sp_load_test"
        
        # Create and train a tokenizer first
        corpus_files = create_larger_test_corpus(test_dir)
        
        config = TokenizerConfig(
            tokenizer_type="sentencepiece",
            vocab_size=200,
            model_type="bpe",
            special_tokens={"unk_token": "<unk>", "pad_token": "<pad>"}
        )
        
        # Train tokenizer
        trainer1 = SentencePieceTokenizerTrainer(config)
        if not trainer1.train(corpus_files, output_dir):
            logger.error("❌ Failed to train tokenizer for loading test")
            return False
        
        # Test original tokenizer
        test_text = "This is a test for loading functionality."
        original_tokens = trainer1.encode(test_text)
        
        # Create new trainer and load the model
        trainer2 = SentencePieceTokenizerTrainer(config)
        if not trainer2.load_tokenizer(output_dir):
            logger.error("❌ Failed to load trained tokenizer")
            return False
        
        # Test loaded tokenizer
        loaded_tokens = trainer2.encode(test_text)
        
        # Compare results
        if original_tokens == loaded_tokens:
            logger.info("✅ SentencePiece loading working correctly")
            logger.info(f"  Test text: {test_text}")
            logger.info(f"  Tokens: {original_tokens}")
            logger.info(f"  Decoded: {trainer2.decode(loaded_tokens)}")
            return True
        else:
            logger.error("❌ Loaded tokenizer produces different results")
            logger.error(f"  Original: {original_tokens}")
            logger.error(f"  Loaded: {loaded_tokens}")
            return False


def test_cli_script():
    """Test the CLI script functionality."""
    if not SENTENCEPIECE_AVAILABLE:
        logger.warning("Skipping CLI script test - SentencePiece not available")
        return True
    
    logger.info("Testing SentencePiece CLI script...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Create test corpus
        corpus_files = create_larger_test_corpus(test_dir)
        
        # Test vocab estimation
        from scripts.train_sentencepiece import estimate_vocab_size
        
        recommended_vocab = estimate_vocab_size([str(f) for f in corpus_files], 1000)
        
        if recommended_vocab > 0:
            logger.info(f"✅ Vocab estimation working: recommended {recommended_vocab}")
        else:
            logger.error("❌ Vocab estimation failed")
            return False
        
        # Test configuration creation
        from scripts.train_sentencepiece import create_sentencepiece_config
        import argparse
        
        # Mock args
        class MockArgs:
            vocab_size = 200
            model_type = "bpe"
            pad_token = "<pad>"
            unk_token = "<unk>"
            bos_token = "<s>"
            eos_token = "</s>"
            character_coverage = 0.9995
            split_by_whitespace = True
            max_sentence_length = 4192
            num_threads = 1
            input_sentence_size = 0
            shuffle_input_sentence = True
            hard_vocab_limit = True
            normalization = True
            shrinking_factor = None
            max_sentencepiece_length = None
        
        config = create_sentencepiece_config(MockArgs())
        
        if config.tokenizer_type == "sentencepiece" and config.vocab_size == 200:
            logger.info("✅ CLI configuration creation working")
            return True
        else:
            logger.error("❌ CLI configuration creation failed")
            return False


def main():
    """Run all SentencePiece CLI tests."""
    logger.info("🔍 Testing SentencePiece CLI Integration...")
    
    tests = [
        ("SentencePiece BPE Training", test_sentencepiece_bpe),
        ("SentencePiece Unigram Training", test_sentencepiece_unigram),
        ("SentencePiece Model Loading", test_sentencepiece_loading),
        ("CLI Script Functionality", test_cli_script),
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
    logger.info(f"SENTENCEPIECE CLI TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All SentencePiece CLI tests passed!")
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    main()