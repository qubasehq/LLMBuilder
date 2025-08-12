#!/usr/bin/env python3
"""
Comprehensive tests for ConversionPipeline.

Tests the automated GGUF conversion workflow with multiple quantization levels,
batch processing, error handling, and validation.
"""

import json
import sys
import tempfile
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, MagicMock
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.conversion_pipeline import (
    ConversionPipeline, ConversionConfig, ConversionResult, PipelineResult,
    create_conversion_pipeline
)


def create_test_model(vocab_size: int = 1000, hidden_size: int = 128) -> nn.Module:
    """Create a simple test model for conversion testing."""
    
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size: int, hidden_size: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=8,
                    dim_feedforward=hidden_size * 4,
                    batch_first=True
                ),
                num_layers=2
            )
            self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            x = self.transformer(x)
            return self.lm_head(x)
    
    return SimpleTransformer(vocab_size, hidden_size)


def create_test_checkpoint(model_path: Path, vocab_size: int = 1000) -> dict:
    """Create a test PyTorch checkpoint file."""
    model = create_test_model(vocab_size)
    
    # Create model metadata
    metadata = {
        'model_name': 'Test Conversion Model',
        'architecture': 'transformer',
        'vocab_size': vocab_size,
        'hidden_size': 128,
        'num_layers': 2,
        'num_heads': 8,
        'training_args': {
            'learning_rate': 0.001,
            'batch_size': 32,
            'max_steps': 1000,
        }
    }
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata,
        'epoch': 10,
        'step': 1000,
    }
    
    torch.save(checkpoint, model_path)
    return checkpoint


def create_test_tokenizer(tokenizer_path: Path) -> dict:
    """Create a test tokenizer file."""
    tokenizer_data = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": 0, "content": "<pad>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 1, "content": "<s>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 2, "content": "</s>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 3, "content": "<unk>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "vocab": {f"token_{i}": i + 4 for i in range(996)},
            "merges": []
        }
    }
    
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer_data, f)
    
    return tokenizer_data


def test_conversion_config():
    """Test ConversionConfig creation and validation."""
    logger.info("Testing ConversionConfig...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "model.pt"
        output_path = temp_path / "output"
        tokenizer_path = temp_path / "tokenizer.json"
        
        # Create test files
        input_path.touch()
        tokenizer_path.touch()
        
        # Test valid configuration
        config = ConversionConfig(
            input_checkpoint=input_path,
            output_dir=output_path,
            quantization_levels=["f16", "q8_0", "q4_0"],
            model_name="Test Model",
            tokenizer_path=tokenizer_path,
            validate_output=True,
            max_retries=2
        )
        
        assert config.input_checkpoint == input_path
        assert config.output_dir == output_path
        assert config.quantization_levels == ["f16", "q8_0", "q4_0"]
        assert config.model_name == "Test Model"
        assert config.tokenizer_path == tokenizer_path
        assert config.validate_output is True
        assert config.max_retries == 2
        
        # Test invalid quantization levels
        try:
            ConversionConfig(
                input_checkpoint=input_path,
                output_dir=output_path,
                quantization_levels=["invalid_level"]
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid quantization levels" in str(e)
    
    logger.info("✅ ConversionConfig working correctly")


def test_conversion_pipeline_basic():
    """Test basic ConversionPipeline functionality."""
    logger.info("Testing basic ConversionPipeline...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "model.pt"
        output_path = temp_path / "output"
        tokenizer_path = temp_path / "tokenizer.json"
        
        # Create test files
        create_test_checkpoint(input_path)
        create_test_tokenizer(tokenizer_path)
        
        # Create configuration
        config = ConversionConfig(
            input_checkpoint=input_path,
            output_dir=output_path,
            quantization_levels=["f16"],  # Just one level for basic test
            model_name="Basic Test Model",
            tokenizer_path=tokenizer_path,
            validate_output=False,  # Skip validation for speed
            max_retries=1
        )
        
        # Create pipeline
        pipeline = ConversionPipeline(config)
        
        assert pipeline.config == config
        assert len(pipeline.results) == 0
        assert output_path.exists()  # Should be created by pipeline
    
    logger.info("✅ Basic ConversionPipeline working correctly")


def test_conversion_pipeline_convert_all():
    """Test full conversion pipeline with multiple quantization levels."""
    logger.info("Testing ConversionPipeline convert_all...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "model.pt"
        output_path = temp_path / "output"
        tokenizer_path = temp_path / "tokenizer.json"
        
        # Create test files
        create_test_checkpoint(input_path, vocab_size=100)  # Smaller for faster testing
        create_test_tokenizer(tokenizer_path)
        
        # Create configuration with multiple levels
        config = ConversionConfig(
            input_checkpoint=input_path,
            output_dir=output_path,
            quantization_levels=["f32", "f16"],  # Two levels for testing
            model_name="Multi Level Test",
            tokenizer_path=tokenizer_path,
            validate_output=False,  # Disable validation for testing
            max_retries=1
        )
        
        # Create and run pipeline
        pipeline = ConversionPipeline(config)
        result = pipeline.convert_all()
        
        # Verify results
        assert isinstance(result, PipelineResult)
        assert result.input_checkpoint == input_path
        assert result.output_dir == output_path
        assert len(result.results) == 2
        assert result.total_time > 0
        
        # Check individual results
        f32_result = result.get_result("f32")
        f16_result = result.get_result("f16")
        
        assert f32_result is not None
        assert f16_result is not None
        
        # At least one should succeed (f32 should always work)
        assert result.successful_conversions >= 1
        assert result.success_rate > 0
        
        # Check output files exist for successful conversions
        for conv_result in result.results:
            if conv_result.success:
                assert conv_result.output_path.exists()
                assert conv_result.file_size_mb > 0
                assert conv_result.conversion_time > 0
    
    logger.info("✅ ConversionPipeline convert_all working correctly")


def test_conversion_pipeline_error_handling():
    """Test error handling and retry logic."""
    logger.info("Testing ConversionPipeline error handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "nonexistent.pt"  # File doesn't exist
        output_path = temp_path / "output"
        
        # Create configuration
        config = ConversionConfig(
            input_checkpoint=input_path,
            output_dir=output_path,
            quantization_levels=["f16"],
            max_retries=2
        )
        
        # Create and run pipeline
        pipeline = ConversionPipeline(config)
        result = pipeline.convert_all()
        
        # Should fail gracefully
        assert result.successful_conversions == 0
        assert result.failed_conversions == 1
        assert result.success_rate == 0.0
        
        # Check error result
        f16_result = result.get_result("f16")
        assert f16_result is not None
        assert not f16_result.success
        assert f16_result.error_message is not None
        assert f16_result.file_size_mb == 0.0
    
    logger.info("✅ ConversionPipeline error handling working correctly")


def test_conversion_pipeline_validation():
    """Test output validation functionality."""
    logger.info("Testing ConversionPipeline validation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create pipeline for testing validation
        config = ConversionConfig(
            input_checkpoint=temp_path / "dummy.pt",
            output_dir=temp_path / "output",
            quantization_levels=["f16"]
        )
        pipeline = ConversionPipeline(config)
        
        # Test validation with non-existent file
        assert not pipeline.validate_conversion(temp_path / "nonexistent.gguf")
        
        # Test validation with empty file
        empty_file = temp_path / "empty.gguf"
        empty_file.touch()
        assert not pipeline.validate_conversion(empty_file)
        
        # Test validation with small file
        small_file = temp_path / "small.gguf"
        with open(small_file, 'wb') as f:
            f.write(b"small")
        assert not pipeline.validate_conversion(small_file)
    
    logger.info("✅ ConversionPipeline validation working correctly")


def test_conversion_pipeline_report():
    """Test conversion report generation."""
    logger.info("Testing ConversionPipeline report generation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "model.pt"
        output_path = temp_path / "output"
        report_path = temp_path / "report.json"
        
        # Create test files
        create_test_checkpoint(input_path, vocab_size=50)  # Very small for speed
        
        # Create configuration
        config = ConversionConfig(
            input_checkpoint=input_path,
            output_dir=output_path,
            quantization_levels=["f32"],  # Just one for testing
            model_name="Report Test Model",
            validate_output=False,
            max_retries=1
        )
        
        # Create and run pipeline
        pipeline = ConversionPipeline(config)
        result = pipeline.convert_all()
        
        # Save report
        pipeline.save_report(report_path)
        
        # Verify report file
        assert report_path.exists()
        
        with open(report_path) as f:
            report_data = json.load(f)
        
        # Check report structure
        assert "pipeline_config" in report_data
        assert "pipeline_result" in report_data
        assert "conversion_results" in report_data
        
        # Check config data
        config_data = report_data["pipeline_config"]
        assert config_data["model_name"] == "Report Test Model"
        assert config_data["quantization_levels"] == ["f32"]
        
        # Check result data
        result_data = report_data["pipeline_result"]
        assert "total_time" in result_data
        assert "success_rate" in result_data
        
        # Check conversion results
        conv_results = report_data["conversion_results"]
        assert len(conv_results) == 1
        assert conv_results[0]["quantization_level"] == "f32"
    
    logger.info("✅ ConversionPipeline report generation working correctly")


def test_create_conversion_pipeline_factory():
    """Test factory function for creating ConversionPipeline."""
    logger.info("Testing create_conversion_pipeline factory...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "model.pt"
        output_path = temp_path / "output"
        
        # Test with defaults
        pipeline1 = create_conversion_pipeline(input_path, output_path)
        assert pipeline1.config.quantization_levels == ["f16", "q8_0", "q4_0"]
        assert pipeline1.config.validate_output is True
        
        # Test with custom parameters
        pipeline2 = create_conversion_pipeline(
            input_path, 
            output_path,
            quantization_levels=["f32", "q8_0"],
            model_name="Factory Test",
            validate_output=False
        )
        assert pipeline2.config.quantization_levels == ["f32", "q8_0"]
        assert pipeline2.config.model_name == "Factory Test"
        assert pipeline2.config.validate_output is False
    
    logger.info("✅ create_conversion_pipeline factory working correctly")


def test_pipeline_result_methods():
    """Test PipelineResult helper methods."""
    logger.info("Testing PipelineResult methods...")
    
    # Create test results
    results = [
        ConversionResult("f16", Path("f16.gguf"), True, 10.0, 5.0),
        ConversionResult("q8_0", Path("q8_0.gguf"), True, 8.0, 7.0),
        ConversionResult("q4_0", Path("q4_0.gguf"), False, 0.0, 3.0, error_message="Failed"),
    ]
    
    pipeline_result = PipelineResult(
        input_checkpoint=Path("input.pt"),
        output_dir=Path("output"),
        total_time=15.0,
        successful_conversions=2,
        failed_conversions=1,
        results=results
    )
    
    # Test success rate calculation
    expected_rate = (2 / 3) * 100  # 66.67%
    assert abs(pipeline_result.success_rate - expected_rate) < 0.01
    
    # Test get_result method
    f16_result = pipeline_result.get_result("f16")
    assert f16_result is not None
    assert f16_result.quantization_level == "f16"
    assert f16_result.success is True
    
    q4_0_result = pipeline_result.get_result("q4_0")
    assert q4_0_result is not None
    assert q4_0_result.success is False
    
    nonexistent_result = pipeline_result.get_result("nonexistent")
    assert nonexistent_result is None
    
    logger.info("✅ PipelineResult methods working correctly")


def main():
    """Run all ConversionPipeline tests."""
    logger.info("🔍 Testing Comprehensive ConversionPipeline...")
    
    tests = [
        ("ConversionConfig", test_conversion_config),
        ("Pipeline Basic", test_conversion_pipeline_basic),
        ("Pipeline Convert All", test_conversion_pipeline_convert_all),
        ("Pipeline Error Handling", test_conversion_pipeline_error_handling),
        ("Pipeline Validation", test_conversion_pipeline_validation),
        ("Pipeline Report", test_conversion_pipeline_report),
        ("Factory Function", test_create_conversion_pipeline_factory),
        ("PipelineResult Methods", test_pipeline_result_methods),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"TEST: {test_name}")
        logger.info("=" * 60)
        
        try:
            test_func()
            logger.info(f"✅ {test_name} PASSED")
            passed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED: {e}")
            failed += 1
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("CONVERSION PIPELINE TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Passed: {passed}/{passed + failed}")
    
    if failed == 0:
        logger.info("🎉 All ConversionPipeline tests passed!")
    else:
        logger.error(f"💥 {failed} tests failed!")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)