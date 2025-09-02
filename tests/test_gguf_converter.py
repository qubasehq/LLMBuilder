#!/usr/bin/env python3
"""
Test enhanced GGUF converter with improved metadata handling and validation.
"""

import sys
import tempfile
import json
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from llmbuilder.core.tools.export_gguf import GGUFConverter, ModelMetadata, GGUFTensorInfo, GGUFValidationResult
from loguru import logger


def create_test_model():
    """Create a simple test model for GGUF conversion."""
    # Simple GPT-2 like model structure
    config = {
        'vocab_size': 1000,
        'block_size': 512,
        'n_embd': 256,
        'n_head': 8,
        'n_layer': 4,
    }
    
    # Create state dict with typical transformer weights
    state_dict = {
        'transformer.wte.weight': torch.randn(config['vocab_size'], config['n_embd']),
        'transformer.wpe.weight': torch.randn(config['block_size'], config['n_embd']),
        'transformer.ln_f.weight': torch.ones(config['n_embd']),
        'transformer.ln_f.bias': torch.zeros(config['n_embd']),
        'lm_head.weight': torch.randn(config['vocab_size'], config['n_embd']),
    }
    
    # Add transformer blocks
    for i in range(config['n_layer']):
        prefix = f'transformer.h.{i}'
        state_dict.update({
            f'{prefix}.ln_1.weight': torch.ones(config['n_embd']),
            f'{prefix}.ln_1.bias': torch.zeros(config['n_embd']),
            f'{prefix}.attn.c_attn.weight': torch.randn(config['n_embd'], 3 * config['n_embd']),
            f'{prefix}.attn.c_attn.bias': torch.zeros(3 * config['n_embd']),
            f'{prefix}.attn.c_proj.weight': torch.randn(config['n_embd'], config['n_embd']),
            f'{prefix}.attn.c_proj.bias': torch.zeros(config['n_embd']),
            f'{prefix}.ln_2.weight': torch.ones(config['n_embd']),
            f'{prefix}.ln_2.bias': torch.zeros(config['n_embd']),
            f'{prefix}.mlp.c_fc.weight': torch.randn(config['n_embd'], 4 * config['n_embd']),
            f'{prefix}.mlp.c_fc.bias': torch.zeros(4 * config['n_embd']),
            f'{prefix}.mlp.c_proj.weight': torch.randn(4 * config['n_embd'], config['n_embd']),
            f'{prefix}.mlp.c_proj.bias': torch.zeros(config['n_embd']),
        })
    
    return state_dict, config


def create_test_tokenizer():
    """Create a simple test tokenizer vocabulary."""
    tokenizer = {
        'model': {'type': 'BPE'},
        'vocab': {f'token_{i}': i for i in range(100)},
        'added_tokens': [
            {'content': '<pad>', 'id': 0},
            {'content': '<s>', 'id': 1},
            {'content': '</s>', 'id': 2},
            {'content': '<unk>', 'id': 3},
        ]
    }
    return tokenizer


def test_model_metadata():
    """Test ModelMetadata functionality."""
    logger.info("Testing ModelMetadata...")
    
    # Test basic metadata creation
    metadata = ModelMetadata(
        name="Test Model",
        architecture="gpt2",
        version="1.0",
        author="Test Author",
        description="A test model",
        vocab_size=1000,
        context_length=512,
        embedding_length=256,
        block_count=4,
        head_count=8
    )
    
    # Test GGUF metadata conversion
    gguf_metadata = metadata.to_gguf_metadata()
    
    expected_keys = [
        "general.name", "general.architecture", "general.version",
        "general.author", "general.description"
    ]
    
    for key in expected_keys:
        if key not in gguf_metadata:
            logger.error(f"❌ Missing key in GGUF metadata: {key}")
            return False
    
    if gguf_metadata["general.name"] != "Test Model":
        logger.error("❌ Incorrect name in GGUF metadata")
        return False
    
    logger.info("✅ ModelMetadata working correctly")
    return True


def test_gguf_converter_basic():
    """Test basic GGUF converter functionality."""
    logger.info("Testing basic GGUF converter...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test model
        state_dict, config = create_test_model()
        model_path = temp_path / "test_model.pt"
        
        # Save test model
        torch.save({
            'model': state_dict,
            'config': config,
            'training_stats': {
                'step': 1000,
                'loss': 2.5,
                'training_time': 3600.0
            }
        }, model_path)
        
        # Create metadata
        metadata = ModelMetadata(
            name="Test Model",
            architecture="gpt2",
            version="1.0",
            description="Test model for GGUF conversion"
        )
        
        # Create converter
        output_path = temp_path / "test_model.gguf"
        converter = GGUFConverter(str(model_path), str(output_path), metadata)
        
        # Test model loading
        if not converter.load_model():
            logger.error("❌ Failed to load test model")
            return False
        
        # Check metadata extraction
        if converter.metadata.training_steps != 1000:
            logger.error("❌ Failed to extract training steps")
            return False
        
        if converter.metadata.loss != 2.5:
            logger.error("❌ Failed to extract loss")
            return False
        
        # Check parameter counting
        param_count = converter._count_parameters()
        if param_count == 0:
            logger.error("❌ Parameter count is zero")
            return False
        
        logger.info(f"✅ Model loaded successfully with {param_count:,} parameters")
        return True


def test_gguf_export():
    """Test complete GGUF export process."""
    logger.info("Testing GGUF export process...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test model and tokenizer
        state_dict, config = create_test_model()
        tokenizer = create_test_tokenizer()
        
        model_path = temp_path / "test_model.pt"
        tokenizer_path = temp_path / "tokenizer.json"
        
        # Save test files
        torch.save({'model': state_dict, 'config': config}, model_path)
        with open(tokenizer_path, 'w') as f:
            json.dump(tokenizer, f)
        
        # Create converter
        metadata = ModelMetadata(
            name="Export Test Model",
            architecture="gpt2",
            version="1.0"
        )
        
        output_path = temp_path / "test_export.gguf"
        converter = GGUFConverter(str(model_path), str(output_path), metadata)
        
        # Export model
        success = converter.export_to_gguf(str(tokenizer_path), validate=True)
        
        if not success:
            logger.error("❌ GGUF export failed")
            return False
        
        # Check output file exists
        if not output_path.exists():
            logger.error("❌ Output GGUF file not created")
            return False
        
        # Check file size is reasonable
        file_size = output_path.stat().st_size
        if file_size < 1000:  # Should be at least 1KB
            logger.error(f"❌ Output file too small: {file_size} bytes")
            return False
        
        # Check model info file
        info_path = output_path.with_suffix('.json')
        if info_path.exists():
            with open(info_path) as f:
                model_info = json.load(f)
            
            if 'parameter_count' not in model_info:
                logger.error("❌ Model info missing parameter count")
                return False
        
        logger.info(f"✅ GGUF export successful, file size: {file_size / 1024:.1f} KB")
        return True


def test_gguf_validation():
    """Test GGUF file validation."""
    logger.info("Testing GGUF validation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test model
        state_dict, config = create_test_model()
        model_path = temp_path / "test_model.pt"
        torch.save({'model': state_dict, 'config': config}, model_path)
        
        # Create converter
        metadata = ModelMetadata(name="Validation Test", architecture="gpt2", version="1.0")
        output_path = temp_path / "validation_test.gguf"
        converter = GGUFConverter(str(model_path), str(output_path), metadata)
        
        # Export model
        success = converter.export_to_gguf(validate=False)  # Don't validate during export
        
        if not success:
            logger.error("❌ Failed to export model for validation test")
            return False
        
        # Test validation
        validation_result = converter.validate_gguf_file()
        
        if not validation_result.is_valid:
            logger.error("❌ Validation failed for valid GGUF file")
            for error in validation_result.errors:
                logger.error(f"  - {error}")
            return False
        
        if validation_result.tensor_count == 0:
            logger.error("❌ Validation shows zero tensors")
            return False
        
        # Test validation of non-existent file
        fake_path = temp_path / "nonexistent.gguf"
        converter.output_path = fake_path
        fake_validation = converter.validate_gguf_file()
        
        if fake_validation.is_valid:
            logger.error("❌ Validation should fail for non-existent file")
            return False
        
        logger.info("✅ GGUF validation working correctly")
        return True


def test_metadata_handling():
    """Test comprehensive metadata handling."""
    logger.info("Testing metadata handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test model with comprehensive config
        state_dict, config = create_test_model()
        model_path = temp_path / "metadata_test.pt"
        
        # Add more comprehensive training metadata
        checkpoint = {
            'model': state_dict,
            'config': config,
            'training_stats': {
                'step': 5000,
                'loss': 1.8,
                'training_time': 7200.0
            },
            'epoch': 10,
            'optimizer': {}  # Indicates training checkpoint
        }
        
        torch.save(checkpoint, model_path)
        
        # Create comprehensive metadata
        metadata = ModelMetadata(
            name="Comprehensive Test Model",
            architecture="gpt2",
            version="2.0",
            author="Test Suite",
            description="Model with comprehensive metadata",
            license="MIT",
            url="https://example.com/model",
            training_data="Test Dataset"
        )
        
        output_path = temp_path / "metadata_test.gguf"
        converter = GGUFConverter(str(model_path), str(output_path), metadata)
        
        # Load model and check metadata extraction
        if not converter.load_model():
            logger.error("❌ Failed to load model")
            return False
        
        # Check training metadata extraction
        if converter.metadata.training_steps != 5000:
            logger.error(f"❌ Wrong training steps: {converter.metadata.training_steps}")
            return False
        
        # Create GGUF metadata
        gguf_metadata = converter.create_gguf_metadata()
        
        # Check comprehensive metadata
        expected_metadata = [
            "general.name", "general.architecture", "general.version",
            "general.author", "general.description", "general.license",
            "general.url", "training.data", "training.steps", "training.loss",
            "gpt2.vocab_size", "gpt2.context_length", "gpt2.embedding_length"
        ]
        
        for key in expected_metadata:
            if key not in gguf_metadata:
                logger.error(f"❌ Missing metadata key: {key}")
                return False
        
        # Check specific values
        if gguf_metadata["general.name"] != "Comprehensive Test Model":
            logger.error("❌ Incorrect model name in metadata")
            return False
        
        if gguf_metadata["training.steps"] != 5000:
            logger.error("❌ Incorrect training steps in metadata")
            return False
        
        logger.info("✅ Comprehensive metadata handling working")
        return True


def test_tensor_info_collection():
    """Test tensor information collection."""
    logger.info("Testing tensor info collection...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test model
        state_dict, config = create_test_model()
        model_path = temp_path / "tensor_test.pt"
        torch.save({'model': state_dict, 'config': config}, model_path)
        
        # Create converter
        metadata = ModelMetadata(name="Tensor Test", architecture="gpt2", version="1.0")
        output_path = temp_path / "tensor_test.gguf"
        converter = GGUFConverter(str(model_path), str(output_path), metadata)
        
        # Load model
        if not converter.load_model():
            logger.error("❌ Failed to load model")
            return False
        
        # Collect tensor info
        converter._collect_tensor_info()
        
        if len(converter.tensor_info) == 0:
            logger.error("❌ No tensor info collected")
            return False
        
        # Check tensor info details
        for tensor_info in converter.tensor_info:
            if not tensor_info.name:
                logger.error("❌ Tensor missing name")
                return False
            
            if len(tensor_info.shape) == 0:
                logger.error("❌ Tensor missing shape")
                return False
            
            if tensor_info.size_bytes == 0:
                logger.error("❌ Tensor has zero size")
                return False
        
        # Check specific tensors exist
        tensor_names = [t.name for t in converter.tensor_info]
        expected_tensors = ['transformer.wte.weight', 'transformer.wpe.weight', 'lm_head.weight']
        
        for expected in expected_tensors:
            if expected not in tensor_names:
                logger.error(f"❌ Missing expected tensor: {expected}")
                return False
        
        logger.info(f"✅ Collected info for {len(converter.tensor_info)} tensors")
        return True


def main():
    """Run all GGUF converter tests."""
    logger.info("🔍 Testing Enhanced GGUF Converter...")
    
    tests = [
        ("ModelMetadata", test_model_metadata),
        ("GGUF Converter Basic", test_gguf_converter_basic),
        ("GGUF Export", test_gguf_export),
        ("GGUF Validation", test_gguf_validation),
        ("Metadata Handling", test_metadata_handling),
        ("Tensor Info Collection", test_tensor_info_collection),
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
    logger.info(f"GGUF CONVERTER TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All GGUF converter tests passed!")
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    main()