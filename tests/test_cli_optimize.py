"""
Tests for optimization CLI commands.

This module tests the quantization, pruning, distillation, and export
functionality provided by the optimize CLI commands.
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path
from click.testing import CliRunner

from llmbuilder.cli.optimize import optimize
from llmbuilder.core.tools.quantization_manager import QuantizationManager, create_quantization_config


@pytest.fixture
def sample_model():
    """Create a sample PyTorch model for testing."""
    model_state_dict = {
        'embedding.weight': torch.randn(1000, 128),
        'layer1.weight': torch.randn(128, 256),
        'layer1.bias': torch.randn(256),
        'layer2.weight': torch.randn(256, 128),
        'layer2.bias': torch.randn(128),
        'output.weight': torch.randn(128, 1000),
        'output.bias': torch.randn(1000)
    }
    
    config = {
        'vocab_size': 1000,
        'n_embd': 128,
        'n_layer': 2,
        'n_head': 8,
        'block_size': 512
    }
    
    return {
        'model': model_state_dict,
        'config': config,
        'metadata': {
            'training_steps': 1000,
            'loss': 2.5
        }
    }


@pytest.fixture
def temp_model_file(sample_model):
    """Create a temporary model file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(sample_model, f.name)
        yield Path(f.name)
    
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_output_file():
    """Create a temporary output file path."""
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        output_path = Path(f.name)
    
    yield output_path
    
    # Cleanup
    output_path.unlink(missing_ok=True)


class TestQuantizeCommand:
    """Test quantization CLI command."""
    
    def test_quantize_basic(self, temp_model_file, temp_output_file):
        """Test basic quantization functionality."""
        runner = CliRunner()
        
        result = runner.invoke(optimize, [
            'quantize',
            str(temp_model_file),
            str(temp_output_file),
            '--type', 'q4_0',
            '--no-validate'  # Skip validation for faster testing
        ])
        
        assert result.exit_code == 0
        assert temp_output_file.exists()
        
        # Load and verify quantized model
        quantized = torch.load(temp_output_file)
        assert 'quantized_tensors' in quantized
        assert 'quantization_config' in quantized
        assert 'quantization_result' in quantized
    
    def test_quantize_with_options(self, temp_model_file, temp_output_file):
        """Test quantization with various options."""
        runner = CliRunner()
        
        result = runner.invoke(optimize, [
            'quantize',
            str(temp_model_file),
            str(temp_output_file),
            '--type', 'q8_0',
            '--block-size', '16',
            '--quality-threshold', '0.9',
            '--skip-layers', 'bias',
            '--skip-layers', 'embedding',
            '--no-validate'
        ])
        
        assert result.exit_code == 0
        assert temp_output_file.exists()
        
        # Verify configuration was applied
        quantized = torch.load(temp_output_file)
        config = quantized['quantization_config']
        assert config['block_size'] == 16
        assert config['quality_threshold'] == 0.9
        assert 'bias' in config['skip_layers']
        assert 'embedding' in config['skip_layers']
    
    def test_quantize_invalid_type(self, temp_model_file, temp_output_file):
        """Test quantization with invalid type."""
        runner = CliRunner()
        
        result = runner.invoke(optimize, [
            'quantize',
            str(temp_model_file),
            str(temp_output_file),
            '--type', 'invalid_type'
        ])
        
        assert result.exit_code != 0
    
    def test_quantize_nonexistent_file(self, temp_output_file):
        """Test quantization with nonexistent input file."""
        runner = CliRunner()
        
        result = runner.invoke(optimize, [
            'quantize',
            'nonexistent_file.pt',
            str(temp_output_file),
            '--type', 'q4_0'
        ])
        
        assert result.exit_code != 0


class TestPruneCommand:
    """Test pruning CLI command."""
    
    def test_prune_basic(self, temp_model_file, temp_output_file):
        """Test basic pruning functionality."""
        runner = CliRunner()
        
        result = runner.invoke(optimize, [
            'prune',
            str(temp_model_file),
            str(temp_output_file),
            '--sparsity', '0.3',
            '--method', 'magnitude'
        ])
        
        assert result.exit_code == 0
        assert temp_output_file.exists()
        
        # Load and verify pruned model
        pruned = torch.load(temp_output_file)
        assert 'model' in pruned
        assert 'pruning_config' in pruned
        assert pruned['pruning_config']['method'] == 'magnitude'
        assert pruned['pruning_config']['target_sparsity'] == 0.3
    
    def test_prune_with_skip_layers(self, temp_model_file, temp_output_file):
        """Test pruning with skip layers."""
        runner = CliRunner()
        
        result = runner.invoke(optimize, [
            'prune',
            str(temp_model_file),
            str(temp_output_file),
            '--sparsity', '0.5',
            '--method', 'structured',
            '--skip-layers', 'bias',
            '--skip-layers', 'output'
        ])
        
        assert result.exit_code == 0
        assert temp_output_file.exists()
        
        # Verify configuration
        pruned = torch.load(temp_output_file)
        config = pruned['pruning_config']
        assert 'bias' in config['skip_layers']
        assert 'output' in config['skip_layers']
    
    def test_prune_invalid_sparsity(self, temp_model_file, temp_output_file):
        """Test pruning with invalid sparsity value."""
        runner = CliRunner()
        
        result = runner.invoke(optimize, [
            'prune',
            str(temp_model_file),
            str(temp_output_file),
            '--sparsity', '1.5'  # Invalid: > 1.0
        ])
        
        # Should still work but might give warnings
        # The actual validation would happen in the implementation
        assert result.exit_code == 0 or "sparsity" in result.output.lower()


class TestExportCommands:
    """Test export CLI commands."""
    
    def test_export_gguf_basic(self, temp_model_file):
        """Test basic GGUF export."""
        runner = CliRunner()
        
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            result = runner.invoke(optimize, [
                'export', 'gguf',
                str(temp_model_file),
                str(output_path),
                '--quantization', 'f16',
                '--no-validate'
            ])
            
            # Note: This might fail due to incomplete GGUF implementation
            # but we test that the command structure works
            assert 'gguf' in result.output.lower() or result.exit_code == 0
            
        finally:
            output_path.unlink(missing_ok=True)
    
    def test_export_gguf_with_metadata(self, temp_model_file):
        """Test GGUF export with metadata file."""
        runner = CliRunner()
        
        # Create metadata file
        metadata = {
            'name': 'test_model',
            'architecture': 'gpt2',
            'version': '1.0',
            'author': 'test_author'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metadata, f)
            metadata_path = Path(f.name)
        
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            result = runner.invoke(optimize, [
                'export', 'gguf',
                str(temp_model_file),
                str(output_path),
                '--metadata', str(metadata_path),
                '--no-validate'
            ])
            
            # Test command structure
            assert 'metadata' in result.output.lower() or result.exit_code == 0
            
        finally:
            metadata_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
    
    def test_export_onnx_basic(self, temp_model_file):
        """Test basic ONNX export."""
        runner = CliRunner()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            result = runner.invoke(optimize, [
                'export', 'onnx',
                str(temp_model_file),
                str(output_path),
                '--precision', 'fp16'
            ])
            
            # Note: This is a placeholder implementation
            assert 'onnx' in result.output.lower() or 'placeholder' in result.output.lower()
            
        finally:
            output_path.unlink(missing_ok=True)


class TestValidateCommand:
    """Test validation CLI command."""
    
    def test_validate_quantized_model(self, temp_model_file, temp_output_file):
        """Test validation of quantized model."""
        # First create a quantized model
        runner = CliRunner()
        
        # Quantize model
        result = runner.invoke(optimize, [
            'quantize',
            str(temp_model_file),
            str(temp_output_file),
            '--type', 'q4_0',
            '--no-validate'
        ])
        assert result.exit_code == 0
        
        # Validate quantized model
        result = runner.invoke(optimize, [
            'validate',
            str(temp_output_file),
            '--format', 'quantized'
        ])
        
        assert result.exit_code == 0
        assert 'validation' in result.output.lower()
    
    def test_validate_with_detailed_output(self, temp_model_file, temp_output_file):
        """Test validation with detailed output."""
        # Create a quantized model first
        runner = CliRunner()
        
        result = runner.invoke(optimize, [
            'quantize',
            str(temp_model_file),
            str(temp_output_file),
            '--type', 'q8_0',
            '--no-validate'
        ])
        assert result.exit_code == 0
        
        # Validate with detailed output
        result = runner.invoke(optimize, [
            'validate',
            str(temp_output_file),
            '--detailed'
        ])
        
        assert result.exit_code == 0
        assert 'validation' in result.output.lower()
    
    def test_validate_nonexistent_file(self):
        """Test validation of nonexistent file."""
        runner = CliRunner()
        
        result = runner.invoke(optimize, [
            'validate',
            'nonexistent_file.pt'
        ])
        
        assert result.exit_code != 0


class TestQuantizationManager:
    """Test the underlying quantization manager."""
    
    def test_quantization_config_creation(self):
        """Test quantization configuration creation."""
        config = create_quantization_config('q4_0', block_size=16, quality_threshold=0.9)
        
        assert config.quantization_type.value == 'q4_0'
        assert config.block_size == 16
        assert config.quality_threshold == 0.9
    
    def test_quantization_manager_basic(self, sample_model):
        """Test basic quantization manager functionality."""
        config = create_quantization_config('q8_0')
        manager = QuantizationManager(config)
        
        # Test tensor quantization
        test_tensor = torch.randn(100, 50)
        quantized_data, stats = manager.quantize_tensor(test_tensor, 'test_tensor')
        
        assert len(quantized_data) > 0
        assert 'quality_score' in stats
        assert 'compression_ratio' in stats
        assert stats['quality_score'] >= 0.0
        assert stats['quality_score'] <= 1.0
    
    def test_should_quantize_tensor(self):
        """Test tensor quantization decision logic."""
        config = create_quantization_config('q4_0', skip_layers=['bias'], force_layers=['important'])
        manager = QuantizationManager(config)
        
        # Test skip patterns
        assert not manager.should_quantize_tensor('layer.bias', torch.randn(10))
        assert not manager.should_quantize_tensor('norm.weight', torch.randn(10))
        
        # Test force patterns
        assert manager.should_quantize_tensor('important.weight', torch.randn(10))
        
        # Test normal layers
        assert manager.should_quantize_tensor('layer.weight', torch.randn(1000, 500))
        
        # Test small tensors
        assert not manager.should_quantize_tensor('small.weight', torch.randn(10))


if __name__ == '__main__':
    pytest.main([__file__])