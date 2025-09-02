"""
Unit tests for PEFT configuration management.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

# Import the PEFT configuration classes
from llmbuilder.core.finetune.peft_config import (
    PEFTConfigManager,
    LoRAConfig,
    QLoRAConfig,
    AdapterConfig,
    AdapterMetadata,
    TARGET_MODULE_MAPPING,
    validate_peft_dependencies,
    get_peft_installation_instructions
)


class TestLoRAConfig:
    """Test LoRA configuration dataclass."""
    
    def test_default_values(self):
        """Test default LoRA configuration values."""
        config = LoRAConfig()
        assert config.r == 16
        assert config.alpha == 32
        assert config.dropout == 0.1
        assert config.target_modules == ["qkv_proj", "out_proj", "fc1", "fc2"]
        assert config.bias == "none"
        assert config.task_type == "CAUSAL_LM"
    
    def test_custom_values(self):
        """Test custom LoRA configuration values."""
        config = LoRAConfig(
            r=32,
            alpha=64,
            dropout=0.2,
            target_modules=["custom_module"],
            bias="all",
            task_type="SEQ_CLS"
        )
        assert config.r == 32
        assert config.alpha == 64
        assert config.dropout == 0.2
        assert config.target_modules == ["custom_module"]
        assert config.bias == "all"
        assert config.task_type == "SEQ_CLS"


class TestQLoRAConfig:
    """Test QLoRA configuration dataclass."""
    
    def test_default_values(self):
        """Test default QLoRA configuration values."""
        config = QLoRAConfig()
        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_use_double_quant is True
        assert config.bnb_4bit_compute_dtype == "float16"
    
    def test_custom_values(self):
        """Test custom QLoRA configuration values."""
        config = QLoRAConfig(
            load_in_4bit=False,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype="bfloat16"
        )
        assert config.load_in_4bit is False
        assert config.bnb_4bit_quant_type == "fp4"
        assert config.bnb_4bit_use_double_quant is False
        assert config.bnb_4bit_compute_dtype == "bfloat16"


class TestAdapterConfig:
    """Test adapter configuration dataclass."""
    
    def test_default_values(self):
        """Test default adapter configuration values."""
        config = AdapterConfig()
        assert config.save_directory == "exports/adapters"
        assert config.merge_and_unload is False
        assert config.auto_save_interval == 500


class TestPEFTConfigManager:
    """Test PEFT configuration manager."""
    
    def get_valid_config(self):
        """Get a valid PEFT configuration for testing."""
        return {
            "peft": {
                "enabled": True,
                "method": "lora",
                "lora": {
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.1,
                    "target_modules": ["qkv_proj", "out_proj"],
                    "bias": "none",
                    "task_type": "CAUSAL_LM"
                },
                "qlora": {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_compute_dtype": "float16"
                },
                "adapter": {
                    "save_directory": "exports/adapters",
                    "merge_and_unload": False,
                    "auto_save_interval": 500
                }
            }
        }
    
    @patch('finetune.peft_config.PEFT_AVAILABLE', True)
    def test_valid_configuration(self):
        """Test valid PEFT configuration."""
        config = self.get_valid_config()
        manager = PEFTConfigManager(config)
        
        assert manager.is_enabled() is True
        assert manager.get_method() == "lora"
        assert manager.lora_config.r == 16
        assert manager.lora_config.alpha == 32
    
    @patch('finetune.peft_config.PEFT_AVAILABLE', False)
    def test_peft_not_available_disabled(self):
        """Test PEFT configuration when libraries not available but disabled."""
        config = {"peft": {"enabled": False}}
        manager = PEFTConfigManager(config)
        assert manager.is_enabled() is False
    
    @patch('finetune.peft_config.PEFT_AVAILABLE', False)
    def test_peft_not_available_enabled_raises_error(self):
        """Test PEFT configuration when libraries not available but enabled."""
        config = {"peft": {"enabled": True}}
        with pytest.raises(ImportError, match="PEFT libraries not available"):
            PEFTConfigManager(config)
    
    @patch('finetune.peft_config.PEFT_AVAILABLE', True)
    def test_invalid_method(self):
        """Test invalid PEFT method."""
        config = self.get_valid_config()
        config["peft"]["method"] = "invalid_method"
        
        with pytest.raises(ValueError, match="Invalid PEFT method"):
            PEFTConfigManager(config)
    
    @patch('finetune.peft_config.PEFT_AVAILABLE', True)
    def test_invalid_lora_rank(self):
        """Test invalid LoRA rank values."""
        config = self.get_valid_config()
        
        # Test rank too low
        config["peft"]["lora"]["r"] = 0
        with pytest.raises(ValueError, match="LoRA rank.*must be an integer between 1 and 256"):
            PEFTConfigManager(config)
        
        # Test rank too high
        config["peft"]["lora"]["r"] = 300
        with pytest.raises(ValueError, match="LoRA rank.*must be an integer between 1 and 256"):
            PEFTConfigManager(config)
    
    @patch('finetune.peft_config.PEFT_AVAILABLE', True)
    def test_invalid_lora_alpha(self):
        """Test invalid LoRA alpha values."""
        config = self.get_valid_config()
        config["peft"]["lora"]["alpha"] = -1
        
        with pytest.raises(ValueError, match="LoRA alpha must be a positive number"):
            PEFTConfigManager(config)
    
    @patch('finetune.peft_config.PEFT_AVAILABLE', True)
    def test_invalid_lora_dropout(self):
        """Test invalid LoRA dropout values."""
        config = self.get_valid_config()
        
        # Test dropout too low
        config["peft"]["lora"]["dropout"] = -0.1
        with pytest.raises(ValueError, match="LoRA dropout must be between 0 and 1"):
            PEFTConfigManager(config)
        
        # Test dropout too high
        config["peft"]["lora"]["dropout"] = 1.5
        with pytest.raises(ValueError, match="LoRA dropout must be between 0 and 1"):
            PEFTConfigManager(config)
    
    @patch('finetune.peft_config.PEFT_AVAILABLE', True)
    def test_invalid_lora_bias(self):
        """Test invalid LoRA bias values."""
        config = self.get_valid_config()
        config["peft"]["lora"]["bias"] = "invalid_bias"
        
        with pytest.raises(ValueError, match="LoRA bias must be"):
            PEFTConfigManager(config)
    
    @patch('finetune.peft_config.PEFT_AVAILABLE', True)
    def test_invalid_task_type(self):
        """Test invalid task type."""
        config = self.get_valid_config()
        config["peft"]["lora"]["task_type"] = "INVALID_TASK"
        
        with pytest.raises(ValueError, match="Task type must be one of"):
            PEFTConfigManager(config)
    
    @patch('finetune.peft_config.PEFT_AVAILABLE', True)
    def test_invalid_target_modules(self):
        """Test invalid target modules."""
        config = self.get_valid_config()
        config["peft"]["lora"]["target_modules"] = "not_a_list"
        
        with pytest.raises(ValueError, match="target_modules must be a list"):
            PEFTConfigManager(config)
    
    @patch('finetune.peft_config.PEFT_AVAILABLE', True)
    def test_qlora_validation(self):
        """Test QLoRA configuration validation."""
        config = self.get_valid_config()
        config["peft"]["method"] = "qlora"
        
        # Test invalid quantization type
        config["peft"]["qlora"]["bnb_4bit_quant_type"] = "invalid_type"
        with pytest.raises(ValueError, match="Quantization type must be"):
            PEFTConfigManager(config)
        
        # Test invalid compute dtype
        config["peft"]["qlora"]["bnb_4bit_quant_type"] = "nf4"  # Reset to valid
        config["peft"]["qlora"]["bnb_4bit_compute_dtype"] = "invalid_dtype"
        with pytest.raises(ValueError, match="Compute dtype must be one of"):
            PEFTConfigManager(config)
    
    @patch('finetune.peft_config.PEFT_AVAILABLE', True)
    def test_adapter_config_validation(self):
        """Test adapter configuration validation."""
        config = self.get_valid_config()
        
        # Test invalid save directory type
        config["peft"]["adapter"]["save_directory"] = 123
        with pytest.raises(ValueError, match="save_directory must be a string"):
            PEFTConfigManager(config)
        
        # Test invalid auto save interval
        config["peft"]["adapter"]["save_directory"] = "valid_dir"  # Reset
        config["peft"]["adapter"]["auto_save_interval"] = -1
        with pytest.raises(ValueError, match="auto_save_interval must be a positive integer"):
            PEFTConfigManager(config)
    
    @patch('finetune.peft_config.PEFT_AVAILABLE', True)
    def test_get_target_modules_expansion(self):
        """Test target module expansion from predefined mappings."""
        config = self.get_valid_config()
        config["peft"]["lora"]["target_modules"] = ["attention"]
        
        manager = PEFTConfigManager(config)
        target_modules = manager.get_target_modules()
        
        expected = TARGET_MODULE_MAPPING["attention"]
        assert target_modules == expected
    
    @patch('finetune.peft_config.PEFT_AVAILABLE', True)
    def test_is_qlora_enabled(self):
        """Test QLoRA enabled detection."""
        config = self.get_valid_config()
        
        # Test LoRA method
        config["peft"]["method"] = "lora"
        manager = PEFTConfigManager(config)
        assert manager.is_qlora_enabled() is False
        
        # Test QLoRA method
        config["peft"]["method"] = "qlora"
        manager = PEFTConfigManager(config)
        assert manager.is_qlora_enabled() is True
    
    @patch('finetune.peft_config.PEFT_AVAILABLE', True)
    @patch('finetune.peft_config.LoraConfig')
    @patch('finetune.peft_config.TaskType')
    def test_get_lora_config(self, mock_task_type, mock_lora_config):
        """Test LoRA config creation for PEFT library."""
        config = self.get_valid_config()
        manager = PEFTConfigManager(config)
        
        # Mock TaskType enum
        mock_task_type.CAUSAL_LM = "CAUSAL_LM"
        
        result = manager.get_lora_config()
        
        # Verify LoraConfig was called with correct parameters
        mock_lora_config.assert_called_once()
        call_args = mock_lora_config.call_args[1]
        assert call_args["r"] == 16
        assert call_args["lora_alpha"] == 32
        assert call_args["lora_dropout"] == 0.1
    
    @patch('finetune.peft_config.PEFT_AVAILABLE', True)
    @patch('finetune.peft_config.BitsAndBytesConfig')
    def test_get_bnb_config(self, mock_bnb_config):
        """Test BitsAndBytesConfig creation."""
        config = self.get_valid_config()
        config["peft"]["method"] = "qlora"
        manager = PEFTConfigManager(config)
        
        with patch('torch.float16') as mock_float16:
            result = manager.get_bnb_config()
            
            # Verify BitsAndBytesConfig was called
            mock_bnb_config.assert_called_once()
            call_args = mock_bnb_config.call_args[1]
            assert call_args["load_in_4bit"] is True
            assert call_args["bnb_4bit_quant_type"] == "nf4"
    
    def test_adapter_metadata_operations(self):
        """Test adapter metadata save and load operations."""
        config = self.get_valid_config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update config to use temp directory
            config["peft"]["adapter"]["save_directory"] = temp_dir
            manager = PEFTConfigManager(config)
            
            # Create test metadata
            lora_config = LoRAConfig(r=16, alpha=32)
            metadata = AdapterMetadata(
                name="test_adapter",
                base_model="test_model",
                creation_date="2024-01-01",
                training_steps=1000,
                lora_config=lora_config,
                performance_metrics={"loss": 0.5}
            )
            
            # Save metadata
            manager.save_adapter_metadata("test_adapter", metadata)
            
            # Verify file was created
            metadata_path = Path(temp_dir) / "test_adapter" / "adapter_metadata.json"
            assert metadata_path.exists()
            
            # Load metadata
            loaded_metadata = manager.load_adapter_metadata("test_adapter")
            assert loaded_metadata is not None
            assert loaded_metadata.name == "test_adapter"
            assert loaded_metadata.base_model == "test_model"
            assert loaded_metadata.training_steps == 1000
            assert loaded_metadata.lora_config.r == 16
    
    def test_load_nonexistent_adapter_metadata(self):
        """Test loading metadata for nonexistent adapter."""
        config = self.get_valid_config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config["peft"]["adapter"]["save_directory"] = temp_dir
            manager = PEFTConfigManager(config)
            
            # Try to load nonexistent adapter
            metadata = manager.load_adapter_metadata("nonexistent_adapter")
            assert metadata is None


class TestTargetModuleMapping:
    """Test target module mapping functionality."""
    
    def test_target_module_mapping_keys(self):
        """Test that target module mapping has expected keys."""
        expected_keys = ["attention", "mlp", "all_linear", "custom"]
        assert all(key in TARGET_MODULE_MAPPING for key in expected_keys)
    
    def test_attention_modules(self):
        """Test attention module mapping."""
        attention_modules = TARGET_MODULE_MAPPING["attention"]
        assert "qkv_proj" in attention_modules
        assert "out_proj" in attention_modules
    
    def test_mlp_modules(self):
        """Test MLP module mapping."""
        mlp_modules = TARGET_MODULE_MAPPING["mlp"]
        assert "fc1" in mlp_modules
        assert "fc2" in mlp_modules
    
    def test_all_linear_modules(self):
        """Test all linear module mapping."""
        all_linear = TARGET_MODULE_MAPPING["all_linear"]
        attention_modules = TARGET_MODULE_MAPPING["attention"]
        mlp_modules = TARGET_MODULE_MAPPING["mlp"]
        
        # All linear should include both attention and MLP modules
        for module in attention_modules + mlp_modules:
            assert module in all_linear


class TestUtilityFunctions:
    """Test utility functions."""
    
    @patch('finetune.peft_config.PEFT_AVAILABLE', True)
    def test_validate_peft_dependencies_available(self):
        """Test dependency validation when available."""
        with patch('importlib.import_module') as mock_import:
            result = validate_peft_dependencies()
            assert result is True
    
    @patch('finetune.peft_config.PEFT_AVAILABLE', False)
    def test_validate_peft_dependencies_unavailable(self):
        """Test dependency validation when unavailable."""
        result = validate_peft_dependencies()
        assert result is False
    
    def test_get_installation_instructions(self):
        """Test installation instructions."""
        instructions = get_peft_installation_instructions()
        assert "pip install" in instructions
        assert "peft" in instructions
        assert "bitsandbytes" in instructions
        assert "accelerate" in instructions


if __name__ == "__main__":
    pytest.main([__file__])