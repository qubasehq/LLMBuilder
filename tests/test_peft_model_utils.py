"""
Unit tests for PEFT model utilities.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock

from llmbuilder.core.finetune.peft_model_utils import (
    ModelArchitectureInspector,
    PEFTTargetModuleValidator,
    validate_model_for_peft,
    get_recommended_lora_rank
)
from llmbuilder.core.finetune.peft_config import PEFTConfigManager, LoRAConfig


class SimpleTestModel(nn.Module):
    """Simple test model for testing PEFT utilities."""
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 256)
        self.transformer = nn.ModuleDict({
            'attention': nn.ModuleDict({
                'qkv_proj': nn.Linear(256, 768),
                'out_proj': nn.Linear(256, 256)
            }),
            'mlp': nn.ModuleDict({
                'fc1': nn.Linear(256, 1024),
                'fc2': nn.Linear(1024, 256)
            })
        })
        self.lm_head = nn.Linear(256, 1000)
    
    def forward(self, x):
        return x


class TestModelArchitectureInspector:
    """Test model architecture inspector."""
    
    def setup_method(self):
        """Setup test model."""
        self.model = SimpleTestModel()
        self.inspector = ModelArchitectureInspector(self.model)
    
    def test_get_all_module_names(self):
        """Test getting all module names."""
        module_names = self.inspector.get_all_module_names()
        
        expected_names = {
            'embedding',
            'transformer',
            'transformer.attention',
            'transformer.attention.qkv_proj',
            'transformer.attention.out_proj',
            'transformer.mlp',
            'transformer.mlp.fc1',
            'transformer.mlp.fc2',
            'lm_head'
        }
        
        assert expected_names.issubset(module_names)
    
    def test_get_linear_module_names(self):
        """Test getting linear module names."""
        linear_modules = self.inspector.get_linear_module_names()
        
        expected_linear = {
            'transformer.attention.qkv_proj',
            'transformer.attention.out_proj',
            'transformer.mlp.fc1',
            'transformer.mlp.fc2',
            'lm_head'
        }
        
        assert expected_linear.issubset(linear_modules)
        # Embedding should not be included (it's not Linear)
        assert 'embedding' not in linear_modules
    
    def test_validate_target_modules_valid(self):
        """Test validation of valid target modules."""
        target_modules = ['transformer.attention.qkv_proj', 'transformer.mlp.fc1']
        results = self.inspector.validate_target_modules(target_modules)
        
        assert all(results.values())
        assert len(results) == 2
    
    def test_validate_target_modules_invalid(self):
        """Test validation of invalid target modules."""
        target_modules = ['nonexistent_module', 'transformer.attention.qkv_proj']
        results = self.inspector.validate_target_modules(target_modules)
        
        assert results['nonexistent_module'] is False
        assert results['transformer.attention.qkv_proj'] is True
    
    def test_validate_target_modules_pattern_match(self):
        """Test validation with pattern matching."""
        target_modules = ['qkv_proj']  # Should match transformer.attention.qkv_proj
        results = self.inspector.validate_target_modules(target_modules)
        
        assert results['qkv_proj'] is True
    
    def test_suggest_target_modules_gpt(self):
        """Test target module suggestions for GPT architecture."""
        suggested = self.inspector.suggest_target_modules("gpt")
        
        # Should include attention and MLP modules
        assert any('qkv_proj' in module for module in suggested)
        assert any('out_proj' in module for module in suggested)
        assert any('fc1' in module for module in suggested)
        assert any('fc2' in module for module in suggested)
    
    def test_get_module_info_linear(self):
        """Test getting info for linear module."""
        info = self.inspector.get_module_info('transformer.attention.qkv_proj')
        
        assert info is not None
        assert info['name'] == 'transformer.attention.qkv_proj'
        assert info['type'] == 'Linear'
        assert info['in_features'] == 256
        assert info['out_features'] == 768
        assert 'parameters' in info
        assert 'trainable_parameters' in info
    
    def test_get_module_info_nonexistent(self):
        """Test getting info for nonexistent module."""
        info = self.inspector.get_module_info('nonexistent_module')
        assert info is None
    
    def test_print_model_structure(self, capsys):
        """Test printing model structure."""
        self.inspector.print_model_structure(max_depth=2)
        captured = capsys.readouterr()
        
        assert "Model Structure:" in captured.out
        assert "transformer" in captured.out


class TestPEFTTargetModuleValidator:
    """Test PEFT target module validator."""
    
    def setup_method(self):
        """Setup test model and config."""
        self.model = SimpleTestModel()
        
        # Create mock config manager
        self.config = {
            "peft": {
                "enabled": True,
                "method": "lora",
                "lora": {
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.1,
                    "target_modules": ["transformer.attention.qkv_proj", "transformer.mlp.fc1"],
                    "bias": "none",
                    "task_type": "CAUSAL_LM"
                }
            }
        }
        
        with patch('finetune.peft_config.PEFT_AVAILABLE', True):
            self.config_manager = PEFTConfigManager(self.config)
        
        self.validator = PEFTTargetModuleValidator(self.model, self.config_manager)
    
    def test_validate_and_fix_target_modules_valid(self):
        """Test validation with valid target modules."""
        target_modules = self.validator.validate_and_fix_target_modules()
        
        expected = ["transformer.attention.qkv_proj", "transformer.mlp.fc1"]
        assert target_modules == expected
    
    def test_validate_and_fix_target_modules_invalid(self):
        """Test validation with invalid target modules."""
        # Modify config to have invalid modules
        self.config["peft"]["lora"]["target_modules"] = ["nonexistent_module"]
        
        with patch('finetune.peft_config.PEFT_AVAILABLE', True):
            config_manager = PEFTConfigManager(self.config)
        
        validator = PEFTTargetModuleValidator(self.model, config_manager)
        
        # Should suggest alternative modules
        target_modules = validator.validate_and_fix_target_modules()
        
        # Should return suggested modules instead of invalid ones
        assert len(target_modules) > 0
        assert "nonexistent_module" not in target_modules
    
    def test_get_target_module_info(self):
        """Test getting target module information."""
        module_info = self.validator.get_target_module_info()
        
        assert len(module_info) == 2
        assert "transformer.attention.qkv_proj" in module_info
        assert "transformer.mlp.fc1" in module_info
        
        qkv_info = module_info["transformer.attention.qkv_proj"]
        assert qkv_info["type"] == "Linear"
        assert qkv_info["in_features"] == 256
        assert qkv_info["out_features"] == 768
    
    def test_estimate_lora_parameters(self):
        """Test LoRA parameter estimation."""
        estimates = self.validator.estimate_lora_parameters()
        
        assert "total_model_parameters" in estimates
        assert "total_lora_parameters" in estimates
        assert "lora_percentage" in estimates
        assert "module_breakdown" in estimates
        
        # Check that LoRA parameters are calculated correctly
        # For qkv_proj: (256 * 16) + (16 * 768) = 4096 + 12288 = 16384
        # For fc1: (256 * 16) + (16 * 1024) = 4096 + 16384 = 20480
        # Total: 36864
        expected_total = 16384 + 20480
        assert estimates["total_lora_parameters"] == expected_total
        
        assert estimates["lora_percentage"] > 0
        assert len(estimates["module_breakdown"]) == 2
    
    def test_print_target_module_summary(self, capsys):
        """Test printing target module summary."""
        self.validator.print_target_module_summary()
        captured = capsys.readouterr()
        
        assert "PEFT Target Module Summary" in captured.out
        assert "Target modules" in captured.out
        assert "Parameter Estimates" in captured.out
        assert "LoRA parameters" in captured.out


class TestUtilityFunctions:
    """Test utility functions."""
    
    def setup_method(self):
        """Setup test model and config."""
        self.model = SimpleTestModel()
        
        self.config = {
            "peft": {
                "enabled": True,
                "method": "lora",
                "lora": {
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.1,
                    "target_modules": ["transformer.attention.qkv_proj"],
                    "bias": "none",
                    "task_type": "CAUSAL_LM"
                }
            }
        }
    
    @patch('finetune.peft_config.PEFT_AVAILABLE', True)
    def test_validate_model_for_peft_valid(self):
        """Test model validation for valid model."""
        config_manager = PEFTConfigManager(self.config)
        result = validate_model_for_peft(self.model, config_manager)
        assert result is True
    
    @patch('finetune.peft_config.PEFT_AVAILABLE', True)
    def test_validate_model_for_peft_no_trainable_params(self):
        """Test model validation with no trainable parameters."""
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        config_manager = PEFTConfigManager(self.config)
        result = validate_model_for_peft(self.model, config_manager)
        assert result is False
    
    @patch('finetune.peft_config.PEFT_AVAILABLE', True)
    def test_validate_model_for_peft_invalid_modules(self):
        """Test model validation with invalid target modules."""
        self.config["peft"]["lora"]["target_modules"] = ["completely_nonexistent_module"]
        
        config_manager = PEFTConfigManager(self.config)
        
        # Mock the validator to raise an exception
        with patch('finetune.peft_model_utils.PEFTTargetModuleValidator') as mock_validator:
            mock_validator.return_value.validate_and_fix_target_modules.side_effect = ValueError("No valid modules")
            
            result = validate_model_for_peft(self.model, config_manager)
            assert result is False
    
    def test_get_recommended_lora_rank_small_model(self):
        """Test LoRA rank recommendation for small model."""
        target_modules = ["transformer.attention.qkv_proj"]  # 256 -> 768
        rank = get_recommended_lora_rank(self.model, target_modules)
        
        # Should recommend 16 for this size (min dim is 256)
        assert rank == 16
    
    def test_get_recommended_lora_rank_no_modules(self):
        """Test LoRA rank recommendation with no valid modules."""
        target_modules = ["nonexistent_module"]
        rank = get_recommended_lora_rank(self.model, target_modules)
        
        # Should return default rank
        assert rank == 16
    
    def test_get_recommended_lora_rank_large_dimensions(self):
        """Test LoRA rank recommendation for larger dimensions."""
        # Create a model with larger dimensions
        large_model = nn.Sequential(
            nn.Linear(2048, 4096),  # Large dimensions
            nn.Linear(4096, 2048)
        )
        
        target_modules = ["0", "1"]  # Module names in Sequential
        rank = get_recommended_lora_rank(large_model, target_modules)
        
        # Should recommend higher rank for larger dimensions
        assert rank >= 64


class TestModelWithDifferentArchitectures:
    """Test with different model architectures."""
    
    def test_conv1d_modules(self):
        """Test that Conv1d modules are detected as linear modules."""
        model = nn.Sequential(
            nn.Conv1d(256, 512, 1),  # 1D convolution (often used as linear layer)
            nn.Linear(512, 256)
        )
        
        inspector = ModelArchitectureInspector(model)
        linear_modules = inspector.get_linear_module_names()
        
        # Both Conv1d and Linear should be detected
        assert "0" in linear_modules  # Conv1d
        assert "1" in linear_modules  # Linear
    
    def test_nested_modules(self):
        """Test with deeply nested modules."""
        model = nn.ModuleDict({
            'level1': nn.ModuleDict({
                'level2': nn.ModuleDict({
                    'level3': nn.Linear(256, 512)
                })
            })
        })
        
        inspector = ModelArchitectureInspector(model)
        all_modules = inspector.get_all_module_names()
        
        assert 'level1.level2.level3' in all_modules
    
    def test_empty_model(self):
        """Test with empty model."""
        model = nn.Module()
        
        inspector = ModelArchitectureInspector(model)
        all_modules = inspector.get_all_module_names()
        linear_modules = inspector.get_linear_module_names()
        
        assert len(all_modules) == 0
        assert len(linear_modules) == 0


if __name__ == "__main__":
    pytest.main([__file__])