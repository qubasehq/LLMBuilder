#!/usr/bin/env python3
"""
Test enhanced TokenizerConfig with validation, serialization, and preset management.
"""

import sys
import tempfile
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from training.train_tokenizer import TokenizerConfig, TokenizerTrainer
from loguru import logger


def test_tokenizer_config_creation():
    """Test basic TokenizerConfig creation and validation."""
    logger.info("Testing TokenizerConfig creation and validation...")
    
    # Test valid configuration
    try:
        config = TokenizerConfig(
            tokenizer_type="huggingface",
            vocab_size=16000,
            model_type="bpe",
            special_tokens={
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>"
            },
            name="Test Config",
            description="A test configuration"
        )
        
        if config.tokenizer_type == "huggingface" and config.vocab_size == 16000:
            logger.info("✅ Valid TokenizerConfig creation working")
        else:
            logger.error("❌ Valid TokenizerConfig creation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Valid TokenizerConfig creation failed: {e}")
        return False
    
    # Test invalid tokenizer type
    try:
        invalid_config = TokenizerConfig(
            tokenizer_type="invalid_type",
            vocab_size=16000,
            model_type="bpe",
            special_tokens={"unk_token": "<unk>"}
        )
        logger.error("❌ Should have failed with invalid tokenizer type")
        return False
    except ValueError as e:
        if "Invalid tokenizer_type" in str(e):
            logger.info("✅ Invalid tokenizer type validation working")
        else:
            logger.error(f"❌ Wrong validation error: {e}")
            return False
    
    # Test invalid vocab size
    try:
        invalid_config = TokenizerConfig(
            tokenizer_type="huggingface",
            vocab_size=50,  # Too small
            model_type="bpe",
            special_tokens={"unk_token": "<unk>"}
        )
        logger.error("❌ Should have failed with invalid vocab size")
        return False
    except ValueError as e:
        if "too small" in str(e):
            logger.info("✅ Invalid vocab size validation working")
        else:
            logger.error(f"❌ Wrong validation error: {e}")
            return False
    
    # Test missing required special tokens
    try:
        invalid_config = TokenizerConfig(
            tokenizer_type="huggingface",
            vocab_size=16000,
            model_type="bpe",
            special_tokens={}  # Missing unk_token
        )
        logger.error("❌ Should have failed with missing unk_token")
        return False
    except ValueError as e:
        if "require 'unk_token'" in str(e):
            logger.info("✅ Missing special token validation working")
        else:
            logger.error(f"❌ Wrong validation error: {e}")
            return False
    
    return True


def test_tokenizer_config_serialization():
    """Test TokenizerConfig serialization and deserialization."""
    logger.info("Testing TokenizerConfig serialization...")
    
    # Create test configuration
    original_config = TokenizerConfig(
        tokenizer_type="sentencepiece",
        vocab_size=32000,
        model_type="unigram",
        special_tokens={
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>"
        },
        name="Serialization Test",
        description="Testing serialization functionality",
        normalization=True,
        lowercase=False,
        pre_tokenizers=["whitespace"],
        trainer_args={
            "character_coverage": 0.9995,
            "split_by_whitespace": True
        }
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_config.json"
        
        # Test saving
        try:
            original_config.save(config_path)
            
            if not config_path.exists():
                logger.error("❌ Config file was not created")
                return False
            
            logger.info("✅ Config saving working")
            
        except Exception as e:
            logger.error(f"❌ Config saving failed: {e}")
            return False
        
        # Test loading
        try:
            loaded_config = TokenizerConfig.load(config_path)
            
            # Compare key attributes
            if (loaded_config.tokenizer_type == original_config.tokenizer_type and
                loaded_config.vocab_size == original_config.vocab_size and
                loaded_config.model_type == original_config.model_type and
                loaded_config.special_tokens == original_config.special_tokens and
                loaded_config.name == original_config.name and
                loaded_config.description == original_config.description):
                
                logger.info("✅ Config loading working")
            else:
                logger.error("❌ Loaded config doesn't match original")
                return False
                
        except Exception as e:
            logger.error(f"❌ Config loading failed: {e}")
            return False
        
        # Test to_dict and from_dict
        try:
            config_dict = original_config.to_dict()
            reconstructed_config = TokenizerConfig.from_dict(config_dict)
            
            if reconstructed_config.tokenizer_type == original_config.tokenizer_type:
                logger.info("✅ Dict serialization working")
            else:
                logger.error("❌ Dict serialization failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Dict serialization failed: {e}")
            return False
    
    return True


def test_tokenizer_config_presets():
    """Test TokenizerConfig preset system."""
    logger.info("Testing TokenizerConfig presets...")
    
    # Test loading presets
    try:
        presets = TokenizerTrainer.load_presets()
        
        if len(presets) == 0:
            logger.error("❌ No presets found")
            return False
        
        logger.info(f"✅ Found {len(presets)} presets")
        
        # Test some key presets
        expected_presets = ["gpt2_small", "bert_base", "sentencepiece_bpe_small"]
        
        for preset_name in expected_presets:
            if preset_name not in presets:
                logger.error(f"❌ Missing expected preset: {preset_name}")
                return False
            
            config = presets[preset_name]
            
            # Validate preset configuration
            if not config.name or not config.description:
                logger.error(f"❌ Preset {preset_name} missing name or description")
                return False
            
            # Test that preset is valid
            try:
                config.validate()
                logger.info(f"✅ Preset {preset_name} is valid")
            except Exception as e:
                logger.error(f"❌ Preset {preset_name} validation failed: {e}")
                return False
        
    except Exception as e:
        logger.error(f"❌ Preset loading failed: {e}")
        return False
    
    # Test preset retrieval by name
    try:
        gpt2_config = TokenizerTrainer.get_preset("gpt2_small")
        
        if gpt2_config.name == "GPT-2 Small":
            logger.info("✅ Preset retrieval by name working")
        else:
            logger.error("❌ Preset retrieval returned wrong config")
            return False
            
    except Exception as e:
        logger.error(f"❌ Preset retrieval failed: {e}")
        return False
    
    # Test invalid preset name
    try:
        invalid_preset = TokenizerTrainer.get_preset("nonexistent_preset")
        logger.error("❌ Should have failed with invalid preset name")
        return False
    except ValueError as e:
        if "Unknown preset" in str(e):
            logger.info("✅ Invalid preset name handling working")
        else:
            logger.error(f"❌ Wrong error for invalid preset: {e}")
            return False
    
    return True


def test_tokenizer_config_compatibility():
    """Test TokenizerConfig compatibility checking."""
    logger.info("Testing TokenizerConfig compatibility...")
    
    # Create two compatible configurations
    config1 = TokenizerConfig(
        tokenizer_type="huggingface",
        vocab_size=16000,
        model_type="bpe",
        special_tokens={"unk_token": "<unk>", "pad_token": "<pad>"}
    )
    
    config2 = TokenizerConfig(
        tokenizer_type="huggingface",
        vocab_size=16000,
        model_type="bpe",
        special_tokens={"unk_token": "<unk>", "pad_token": "<pad>"},
        name="Different Name"  # Different name but same core config
    )
    
    # Test compatibility
    if config1.is_compatible_with(config2):
        logger.info("✅ Compatible configurations detected correctly")
    else:
        logger.error("❌ Compatible configurations not detected")
        return False
    
    # Create incompatible configuration
    config3 = TokenizerConfig(
        tokenizer_type="sentencepiece",  # Different type
        vocab_size=16000,
        model_type="bpe",
        special_tokens={"unk_token": "<unk>", "pad_token": "<pad>"}
    )
    
    # Test incompatibility
    if not config1.is_compatible_with(config3):
        logger.info("✅ Incompatible configurations detected correctly")
    else:
        logger.error("❌ Incompatible configurations not detected")
        return False
    
    return True


def test_tokenizer_config_copy():
    """Test TokenizerConfig copying with overrides."""
    logger.info("Testing TokenizerConfig copying...")
    
    original_config = TokenizerConfig(
        tokenizer_type="huggingface",
        vocab_size=16000,
        model_type="bpe",
        special_tokens={"unk_token": "<unk>"},
        name="Original"
    )
    
    # Test copy without overrides
    try:
        copied_config = original_config.copy()
        
        if (copied_config.tokenizer_type == original_config.tokenizer_type and
            copied_config.vocab_size == original_config.vocab_size and
            copied_config.name == original_config.name):
            
            logger.info("✅ Config copying without overrides working")
        else:
            logger.error("❌ Config copying without overrides failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Config copying failed: {e}")
        return False
    
    # Test copy with overrides
    try:
        modified_config = original_config.copy(
            vocab_size=32000,
            name="Modified"
        )
        
        if (modified_config.tokenizer_type == original_config.tokenizer_type and
            modified_config.vocab_size == 32000 and
            modified_config.name == "Modified"):
            
            logger.info("✅ Config copying with overrides working")
        else:
            logger.error("❌ Config copying with overrides failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Config copying with overrides failed: {e}")
        return False
    
    return True


def test_tokenizer_config_summary():
    """Test TokenizerConfig summary generation."""
    logger.info("Testing TokenizerConfig summary...")
    
    config = TokenizerConfig(
        tokenizer_type="sentencepiece",
        vocab_size=32000,
        model_type="unigram",
        special_tokens={
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>"
        },
        name="Test Summary Config",
        description="Configuration for testing summary generation",
        normalization=True,
        lowercase=False,
        pre_tokenizers=["whitespace", "punctuation"]
    )
    
    try:
        summary = config.get_summary()
        
        # Check that summary contains key information
        required_info = [
            "Test Summary Config",
            "sentencepiece",
            "unigram",
            "32,000",
            "4",  # number of special tokens
            "whitespace, punctuation",
            "enabled"  # normalization
        ]
        
        for info in required_info:
            if info not in summary:
                logger.error(f"❌ Summary missing required info: {info}")
                logger.error(f"Summary: {summary}")
                return False
        
        logger.info("✅ Config summary generation working")
        logger.info(f"Sample summary:\n{summary}")
        
    except Exception as e:
        logger.error(f"❌ Config summary generation failed: {e}")
        return False
    
    return True


def test_sentencepiece_specific_validation():
    """Test SentencePiece-specific validation."""
    logger.info("Testing SentencePiece-specific validation...")
    
    # Test valid SentencePiece configuration
    try:
        valid_config = TokenizerConfig(
            tokenizer_type="sentencepiece",
            vocab_size=16000,
            model_type="bpe",
            special_tokens={"unk_token": "<unk>"},
            trainer_args={
                "character_coverage": 0.9995,
                "max_sentence_length": 4192,
                "num_threads": 1
            }
        )
        logger.info("✅ Valid SentencePiece config creation working")
        
    except Exception as e:
        logger.error(f"❌ Valid SentencePiece config creation failed: {e}")
        return False
    
    # Test invalid character coverage
    try:
        invalid_config = TokenizerConfig(
            tokenizer_type="sentencepiece",
            vocab_size=16000,
            model_type="bpe",
            special_tokens={"unk_token": "<unk>"},
            trainer_args={"character_coverage": 1.5}  # Invalid > 1.0
        )
        logger.error("❌ Should have failed with invalid character_coverage")
        return False
    except ValueError as e:
        if "character_coverage" in str(e):
            logger.info("✅ Invalid character_coverage validation working")
        else:
            logger.error(f"❌ Wrong validation error: {e}")
            return False
    
    # Test invalid model type for SentencePiece
    try:
        invalid_config = TokenizerConfig(
            tokenizer_type="sentencepiece",
            vocab_size=16000,
            model_type="word",  # Not supported by SentencePiece
            special_tokens={"unk_token": "<unk>"}
        )
        logger.error("❌ Should have failed with unsupported model type")
        return False
    except ValueError as e:
        if "does not support" in str(e):
            logger.info("✅ Invalid SentencePiece model type validation working")
        else:
            logger.error(f"❌ Wrong validation error: {e}")
            return False
    
    return True


def test_huggingface_specific_validation():
    """Test HuggingFace-specific validation."""
    logger.info("Testing HuggingFace-specific validation...")
    
    # Test valid HuggingFace configuration
    try:
        valid_config = TokenizerConfig(
            tokenizer_type="huggingface",
            vocab_size=16000,
            model_type="bpe",
            special_tokens={"unk_token": "<unk>"},
            pre_tokenizers=["whitespace", "punctuation"],
            post_processors=["bert"],
            trainer_args={
                "min_frequency": 2,
                "continuing_subword_prefix": "##"
            }
        )
        logger.info("✅ Valid HuggingFace config creation working")
        
    except Exception as e:
        logger.error(f"❌ Valid HuggingFace config creation failed: {e}")
        return False
    
    # Test invalid pre-tokenizer
    try:
        invalid_config = TokenizerConfig(
            tokenizer_type="huggingface",
            vocab_size=16000,
            model_type="bpe",
            special_tokens={"unk_token": "<unk>"},
            pre_tokenizers=["invalid_pretokenizer"]
        )
        logger.error("❌ Should have failed with invalid pre-tokenizer")
        return False
    except ValueError as e:
        if "Invalid HuggingFace pre-tokenizer" in str(e):
            logger.info("✅ Invalid pre-tokenizer validation working")
        else:
            logger.error(f"❌ Wrong validation error: {e}")
            return False
    
    # Test invalid model type for HuggingFace
    try:
        invalid_config = TokenizerConfig(
            tokenizer_type="huggingface",
            vocab_size=16000,
            model_type="char",  # Not supported by HuggingFace
            special_tokens={"unk_token": "<unk>"}
        )
        logger.error("❌ Should have failed with unsupported model type")
        return False
    except ValueError as e:
        if "do not support 'char'" in str(e):
            logger.info("✅ Invalid HuggingFace model type validation working")
        else:
            logger.error(f"❌ Wrong validation error: {e}")
            return False
    
    return True


def main():
    """Run all TokenizerConfig tests."""
    logger.info("🔍 Testing TokenizerConfig implementation...")
    
    tests = [
        ("TokenizerConfig Creation", test_tokenizer_config_creation),
        ("TokenizerConfig Serialization", test_tokenizer_config_serialization),
        ("TokenizerConfig Presets", test_tokenizer_config_presets),
        ("TokenizerConfig Compatibility", test_tokenizer_config_compatibility),
        ("TokenizerConfig Copy", test_tokenizer_config_copy),
        ("TokenizerConfig Summary", test_tokenizer_config_summary),
        ("SentencePiece Validation", test_sentencepiece_specific_validation),
        ("HuggingFace Validation", test_huggingface_specific_validation),
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
    logger.info(f"TOKENIZER CONFIG TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All TokenizerConfig tests passed!")
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    main()