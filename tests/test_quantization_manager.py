#!/usr/bin/env python3
"""
Test comprehensive quantization manager with quality validation and optimization.
"""

import sys
import tempfile
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from tools.quantization_manager import (
    QuantizationManager, QuantizationConfig, QuantizationType, 
    QuantizationResult, create_quantization_config
)
from loguru import logger


def create_test_tensors():
    """Create test tensors with different characteristics."""
    tensors = {
        # Large weight matrix (should be quantized)
        'transformer.wte.weight': torch.randn(1000, 256) * 0.1,
        
        # Small bias vector (should be skipped)
        'transformer.bias': torch.randn(256) * 0.01,
        
        # Layer norm weights (should be skipped)
        'transformer.ln_f.weight': torch.ones(256),
        
        # Attention weights (should be quantized)
        'transformer.h.0.attn.c_attn.weight': torch.randn(256, 768) * 0.05,
        
        # MLP weights (should be quantized)
        'transformer.h.0.mlp.c_fc.weight': torch.randn(256, 1024) * 0.08,
        
        # Very small tensor (should be skipped)
        'small_param': torch.randn(10),
    }
    
    return tensors


def test_quantization_config():
    """Test quantization configuration creation and validation."""
    logger.info("Testing quantization configuration...")
    
    # Test basic config creation
    config = QuantizationConfig(
        quantization_type=QuantizationType.Q4_0,
        quality_threshold=0.95,
        skip_layers=['bias', 'norm'],
        force_layers=['important']
    )
    
    if config.quantization_type != QuantizationType.Q4_0:
        logger.error("❌ Quantization type not set correctly")
        return False
    
    if config.quality_threshold != 0.95:
        logger.error("❌ Quality threshold not set correctly")
        return False
    
    # Test factory function
    config2 = create_quantization_config('q8_0', quality_threshold=0.9)
    
    if config2.quantization_type != QuantizationType.Q8_0:
        logger.error("❌ Factory function failed")
        return False
    
    # Test invalid quantization type
    try:
        create_quantization_config('invalid_type')
        logger.error("❌ Should have failed with invalid type")
        return False
    except ValueError:
        pass  # Expected
    
    logger.info("✅ Quantization configuration working correctly")
    return True


def test_tensor_quantization():
    """Test individual tensor quantization methods."""
    logger.info("Testing tensor quantization methods...")
    
    # Create test tensor
    test_tensor = torch.randn(100, 50) * 0.1
    
    # Test different quantization types
    quantization_types = [
        QuantizationType.F32,
        QuantizationType.F16,
        QuantizationType.Q8_0,
        QuantizationType.Q4_0,
        QuantizationType.Q4_1,
    ]
    
    for quant_type in quantization_types:
        config = QuantizationConfig(quantization_type=quant_type)
        manager = QuantizationManager(config)
        
        try:
            quantized_data, stats = manager.quantize_tensor(test_tensor, f"test_{quant_type.value}")
            
            # Check that we got data
            if len(quantized_data) == 0:
                logger.error(f"❌ No quantized data for {quant_type.value}")
                return False
            
            # Check stats
            required_stats = ['quality_score', 'original_size', 'quantized_size', 'compression_ratio']
            for stat in required_stats:
                if stat not in stats:
                    logger.error(f"❌ Missing stat {stat} for {quant_type.value}")
                    return False
            
            # Check quality score is reasonable
            if not (0.0 <= stats['quality_score'] <= 1.0):
                logger.error(f"❌ Invalid quality score for {quant_type.value}: {stats['quality_score']}")
                return False
            
            logger.info(f"✅ {quant_type.value}: quality={stats['quality_score']:.3f}, "
                       f"compression={stats['compression_ratio']:.2f}x")
            
        except Exception as e:
            logger.error(f"❌ Failed to quantize with {quant_type.value}: {e}")
            return False
    
    logger.info("✅ Tensor quantization methods working correctly")
    return True


def test_model_quantization():
    """Test full model quantization."""
    logger.info("Testing full model quantization...")
    
    # Create test model
    test_tensors = create_test_tensors()
    
    # Test with Q4_0 quantization
    config = QuantizationConfig(
        quantization_type=QuantizationType.Q4_0,
        quality_threshold=0.8,
        skip_layers=['bias', 'ln', 'norm']
    )
    
    manager = QuantizationManager(config)
    
    # Track progress
    progress_calls = []
    def progress_callback(message, progress):
        progress_calls.append((message, progress))
    
    manager.set_progress_callback(progress_callback)
    
    # Quantize model
    result = manager.quantize_model(test_tensors)
    
    # Check result
    if not isinstance(result, QuantizationResult):
        logger.error("❌ Invalid result type")
        return False
    
    if result.original_size == 0:
        logger.error("❌ Original size is zero")
        return False
    
    if result.quantized_size == 0:
        logger.error("❌ Quantized size is zero")
        return False
    
    if result.compression_ratio <= 0:
        logger.error("❌ Invalid compression ratio")
        return False
    
    if not (0.0 <= result.quality_score <= 1.0):
        logger.error("❌ Invalid quality score")
        return False
    
    # Check that progress was reported
    if len(progress_calls) == 0:
        logger.error("❌ No progress reported")
        return False
    
    # Check tensor results
    if len(result.tensor_results) != len(test_tensors):
        logger.error("❌ Missing tensor results")
        return False
    
    # Check that some tensors were skipped
    skipped_count = sum(1 for stats in result.tensor_results.values() 
                       if stats.get('skipped', False))
    
    if skipped_count == 0:
        logger.error("❌ No tensors were skipped (bias/norm should be skipped)")
        return False
    
    logger.info(f"✅ Model quantization successful:")
    logger.info(f"  - Compression ratio: {result.compression_ratio:.2f}x")
    logger.info(f"  - Quality score: {result.quality_score:.3f}")
    logger.info(f"  - Size reduction: {result.size_reduction_mb:.2f} MB")
    logger.info(f"  - Skipped tensors: {skipped_count}")
    
    return True

def test_quantization_validation():
    """Test quantization validation and quality analysis."""
    logger.info("Testing quantization validation...")
    
    # Create test model
    test_tensors = create_test_tensors()
    
    # Test with high quality threshold
    config = QuantizationConfig(
        quantization_type=QuantizationType.Q4_0,
        quality_threshold=0.95  # High threshold
    )
    
    manager = QuantizationManager(config)
    result = manager.quantize_model(test_tensors)
    
    # Validate quantization
    validation = manager.validate_quantization(test_tensors, result)
    
    # Check validation structure
    required_keys = [
        'overall_quality', 'compression_ratio', 'size_reduction_mb',
        'low_quality_tensors', 'high_compression_tensors', 
        'validation_passed', 'recommendations'
    ]
    
    for key in required_keys:
        if key not in validation:
            logger.error(f"❌ Missing validation key: {key}")
            return False
    
    # Check that validation provides useful information
    if not isinstance(validation['low_quality_tensors'], list):
        logger.error("❌ low_quality_tensors should be a list")
        return False
    
    if not isinstance(validation['recommendations'], list):
        logger.error("❌ recommendations should be a list")
        return False
    
    logger.info(f"✅ Validation completed:")
    logger.info(f"  - Overall quality: {validation['overall_quality']:.3f}")
    logger.info(f"  - Validation passed: {validation['validation_passed']}")
    logger.info(f"  - Low quality tensors: {len(validation['low_quality_tensors'])}")
    logger.info(f"  - Recommendations: {len(validation['recommendations'])}")
    
    return True


def test_tensor_selection():
    """Test tensor selection logic for quantization."""
    logger.info("Testing tensor selection logic...")
    
    config = QuantizationConfig(
        quantization_type=QuantizationType.Q4_0,
        skip_layers=['bias', 'norm'],
        force_layers=['important']
    )
    
    manager = QuantizationManager(config)
    
    # Test cases
    test_cases = [
        ('transformer.wte.weight', True),  # Should quantize
        ('transformer.bias', False),       # Should skip (bias)
        ('transformer.ln_f.weight', False),  # Should skip (norm)
        ('important.weight', True),        # Should quantize (forced)
        ('important.bias', True),          # Should quantize (forced overrides skip)
        ('small_tensor', False),           # Should skip (too small)
    ]
    
    # Create test tensors
    test_tensors = {
        'transformer.wte.weight': torch.randn(1000, 256),
        'transformer.bias': torch.randn(256),
        'transformer.ln_f.weight': torch.ones(256),
        'important.weight': torch.randn(100, 100),
        'important.bias': torch.randn(100),
        'small_tensor': torch.randn(10),  # Very small
    }
    
    for name, expected in test_cases:
        if name in test_tensors:
            result = manager.should_quantize_tensor(name, test_tensors[name])
            if result != expected:
                logger.error(f"❌ Wrong decision for {name}: got {result}, expected {expected}")
                return False
            
            logger.info(f"✅ {name}: {'quantize' if result else 'skip'}")
    
    logger.info("✅ Tensor selection logic working correctly")
    return True


def test_quality_metrics():
    """Test quality score calculation."""
    logger.info("Testing quality metrics...")
    
    # Create test tensors with known properties
    original = torch.randn(1000)
    
    # Perfect reconstruction
    perfect = original.clone()
    
    # Noisy reconstruction
    noisy = original + torch.randn(1000) * 0.1
    
    # Very noisy reconstruction
    very_noisy = original + torch.randn(1000) * 1.0
    
    config = QuantizationConfig(quantization_type=QuantizationType.Q4_0)
    manager = QuantizationManager(config)
    
    # Test quality scores
    perfect_score = manager._calculate_quality_score(original, perfect)
    noisy_score = manager._calculate_quality_score(original, noisy)
    very_noisy_score = manager._calculate_quality_score(original, very_noisy)
    
    # Check that scores are in correct order
    if not (perfect_score > noisy_score > very_noisy_score):
        logger.error(f"❌ Quality scores not in expected order: "
                    f"perfect={perfect_score:.3f}, noisy={noisy_score:.3f}, "
                    f"very_noisy={very_noisy_score:.3f}")
        return False
    
    # Check that perfect score is close to 1.0
    if perfect_score < 0.99:
        logger.error(f"❌ Perfect reconstruction score too low: {perfect_score:.3f}")
        return False
    
    # Check that all scores are in valid range
    scores = [perfect_score, noisy_score, very_noisy_score]
    for i, score in enumerate(scores):
        if not (0.0 <= score <= 1.0):
            logger.error(f"❌ Score {i} out of range: {score}")
            return False
    
    logger.info(f"✅ Quality metrics working correctly:")
    logger.info(f"  - Perfect: {perfect_score:.3f}")
    logger.info(f"  - Noisy: {noisy_score:.3f}")
    logger.info(f"  - Very noisy: {very_noisy_score:.3f}")
    
    return True


def test_quantization_report():
    """Test quantization report generation."""
    logger.info("Testing quantization report generation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test model and quantize
        test_tensors = create_test_tensors()
        
        config = QuantizationConfig(
            quantization_type=QuantizationType.Q8_0,
            quality_threshold=0.9
        )
        
        manager = QuantizationManager(config)
        result = manager.quantize_model(test_tensors)
        validation = manager.validate_quantization(test_tensors, result)
        
        # Save report
        report_path = temp_path / "quantization_report.json"
        manager.save_quantization_report(result, validation, report_path)
        
        # Check that report was created
        if not report_path.exists():
            logger.error("❌ Report file not created")
            return False
        
        # Check report size is reasonable
        file_size = report_path.stat().st_size
        if file_size < 100:  # Should be at least 100 bytes
            logger.error(f"❌ Report file too small: {file_size} bytes")
            return False
        
        # Try to load and validate report structure
        import json
        try:
            with open(report_path) as f:
                report = json.load(f)
            
            required_sections = [
                'quantization_config', 'quantization_result', 
                'validation_result', 'tensor_details', 'summary'
            ]
            
            for section in required_sections:
                if section not in report:
                    logger.error(f"❌ Missing report section: {section}")
                    return False
            
            # Check summary section
            summary = report['summary']
            if summary['total_tensors'] != len(test_tensors):
                logger.error("❌ Incorrect tensor count in summary")
                return False
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in report: {e}")
            return False
        
        logger.info(f"✅ Quantization report generated successfully")
        logger.info(f"  - File size: {file_size} bytes")
        logger.info(f"  - Total tensors: {summary['total_tensors']}")
        logger.info(f"  - Quantized tensors: {summary['quantized_tensors']}")
        
        return True


def test_compression_ratios():
    """Test that different quantization types achieve expected compression ratios."""
    logger.info("Testing compression ratios...")
    
    # Create a large test tensor
    test_tensor = torch.randn(1000, 1000) * 0.1
    test_model = {'large_weight': test_tensor}
    
    # Test different quantization types and their expected compression ratios
    expected_ratios = {
        QuantizationType.F32: 1.0,    # No compression
        QuantizationType.F16: 2.0,    # Half precision
        QuantizationType.Q8_0: 3.5,   # ~4x with overhead
        QuantizationType.Q4_0: 7.0,   # ~8x with overhead
    }
    
    for quant_type, expected_ratio in expected_ratios.items():
        config = QuantizationConfig(quantization_type=quant_type)
        manager = QuantizationManager(config)
        
        result = manager.quantize_model(test_model)
        
        # Check compression ratio is reasonable
        ratio_diff = abs(result.compression_ratio - expected_ratio)
        tolerance = expected_ratio * 0.3  # 30% tolerance
        
        if ratio_diff > tolerance:
            logger.error(f"❌ {quant_type.value}: compression ratio {result.compression_ratio:.2f}x "
                        f"too far from expected {expected_ratio:.2f}x")
            return False
        
        logger.info(f"✅ {quant_type.value}: {result.compression_ratio:.2f}x compression "
                   f"(expected ~{expected_ratio:.2f}x)")
    
    logger.info("✅ Compression ratios are within expected ranges")
    return True


def main():
    """Run all quantization manager tests."""
    logger.info("🔍 Testing Comprehensive Quantization Manager...")
    
    tests = [
        ("Quantization Config", test_quantization_config),
        ("Tensor Quantization", test_tensor_quantization),
        ("Model Quantization", test_model_quantization),
        ("Quantization Validation", test_quantization_validation),
        ("Tensor Selection", test_tensor_selection),
        ("Quality Metrics", test_quality_metrics),
        ("Quantization Report", test_quantization_report),
        ("Compression Ratios", test_compression_ratios),
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
    logger.info(f"QUANTIZATION MANAGER TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All quantization manager tests passed!")
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    main()