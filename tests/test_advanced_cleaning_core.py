"""
Tests for the core infrastructure of the Advanced Cybersecurity Dataset Cleaning system.

This module tests the base data models, configuration management, and core orchestrator
functionality to ensure the infrastructure is working correctly.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Import the modules we're testing
from training.advanced_cleaning.data_models import (
    Entity, EntityType, CleaningOperation, CleaningOperationType,
    CleaningStats, CleaningResult
)
from training.advanced_cleaning.base_module import CleaningModule
from training.advanced_cleaning.config_manager import (
    ConfigManager, AdvancedCleaningConfig, BoilerplateConfig
)
from training.advanced_cleaning.advanced_text_cleaner import AdvancedTextCleaner


class MockCleaningModule(CleaningModule):
    """Mock cleaning module for testing."""
    
    def _initialize_module(self) -> None:
        """Initialize mock module."""
        self.processed_count = 0
    
    def clean_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, CleaningOperation]:
        """Mock text cleaning that removes 'test' words."""
        self.processed_count += 1
        original_length = len(text)
        cleaned_text = text.replace('test', '')
        final_length = len(cleaned_text)
        
        operation = self._create_operation(
            description="Removed 'test' words",
            original_length=original_length,
            final_length=final_length,
            items_removed=text.count('test'),
            success=True
        )
        
        return cleaned_text, operation
    
    def get_operation_type(self) -> CleaningOperationType:
        """Return mock operation type."""
        return CleaningOperationType.BOILERPLATE_REMOVAL


class TestEntity:
    """Test Entity data model."""
    
    def test_entity_creation(self):
        """Test basic entity creation."""
        entity = Entity(
            text="CVE-2023-1234",
            label="CVE",
            start_pos=10,
            end_pos=22,
            confidence=0.95,
            entity_type=EntityType.CVE
        )
        
        assert entity.text == "CVE-2023-1234"
        assert entity.label == "CVE"
        assert entity.start_pos == 10
        assert entity.end_pos == 22
        assert entity.confidence == 0.95
        assert entity.entity_type == EntityType.CVE
    
    def test_entity_validation(self):
        """Test entity validation."""
        # Test invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            Entity("test", "TEST", 0, 4, 1.5, EntityType.OTHER)
        
        # Test invalid positions
        with pytest.raises(ValueError, match="Position values must be non-negative"):
            Entity("test", "TEST", -1, 4, 0.9, EntityType.OTHER)
        
        with pytest.raises(ValueError, match="start_pos must be less than end_pos"):
            Entity("test", "TEST", 5, 4, 0.9, EntityType.OTHER)
    
    def test_entity_serialization(self):
        """Test entity to/from dict conversion."""
        entity = Entity(
            text="firewall",
            label="TOOL",
            start_pos=0,
            end_pos=8,
            confidence=0.8,
            entity_type=EntityType.TOOL,
            metadata={"source": "test"}
        )
        
        entity_dict = entity.to_dict()
        assert entity_dict["text"] == "firewall"
        assert entity_dict["entity_type"] == "TOOL"
        assert entity_dict["metadata"]["source"] == "test"
        
        # Test round-trip conversion
        restored_entity = Entity.from_dict(entity_dict)
        assert restored_entity.text == entity.text
        assert restored_entity.entity_type == entity.entity_type
        assert restored_entity.metadata == entity.metadata


class TestCleaningOperation:
    """Test CleaningOperation data model."""
    
    def test_operation_creation(self):
        """Test basic operation creation."""
        operation = CleaningOperation(
            operation_type=CleaningOperationType.BOILERPLATE_REMOVAL,
            module_name="TestModule",
            description="Test operation",
            original_length=100,
            final_length=80,
            items_removed=5,
            processing_time=0.1,
            success=True
        )
        
        assert operation.operation_type == CleaningOperationType.BOILERPLATE_REMOVAL
        assert operation.module_name == "TestModule"
        assert operation.characters_removed == 20
        assert operation.reduction_percentage == 20.0
        assert operation.success is True
    
    def test_operation_calculations(self):
        """Test operation calculations."""
        operation = CleaningOperation(
            operation_type=CleaningOperationType.QUALITY_ASSESSMENT,
            module_name="QualityModule",
            description="Quality check",
            original_length=200,
            final_length=150
        )
        
        assert operation.characters_removed == 50
        assert operation.reduction_percentage == 25.0
    
    def test_operation_serialization(self):
        """Test operation serialization."""
        operation = CleaningOperation(
            operation_type=CleaningOperationType.LANGUAGE_FILTERING,
            module_name="LanguageFilter",
            description="Language filtering",
            original_length=100,
            final_length=90,
            metadata={"language": "en", "confidence": 0.95}
        )
        
        op_dict = operation.to_dict()
        assert op_dict["operation_type"] == "language_filtering"
        assert op_dict["characters_removed"] == 10
        assert op_dict["metadata"]["language"] == "en"


class TestCleaningStats:
    """Test CleaningStats data model."""
    
    def test_stats_creation(self):
        """Test basic stats creation."""
        stats = CleaningStats(
            original_length=1000,
            final_length=800,
            boilerplate_removed=50,
            quality_score=0.85,
            entities_count=5,
            processing_modules=["Module1", "Module2"],
            operations_performed=3,
            errors_encountered=1
        )
        
        assert stats.characters_removed == 200
        assert stats.reduction_percentage == 20.0
        assert stats.success_rate == 2/3  # (3-1)/3
    
    def test_stats_serialization(self):
        """Test stats serialization."""
        stats = CleaningStats(
            original_length=500,
            final_length=400,
            quality_score=0.9
        )
        
        stats_dict = stats.to_dict()
        assert stats_dict["original_length"] == 500
        assert stats_dict["reduction_percentage"] == 20.0
        assert stats_dict["success_rate"] == 1.0


class TestCleaningResult:
    """Test CleaningResult data model."""
    
    def test_result_creation(self):
        """Test basic result creation."""
        result = CleaningResult(
            original_text="This is a test text with test words",
            cleaned_text="This is a  text with  words",
            quality_score=0.9,
            success=True
        )
        
        assert result.characters_removed == 8  # "test" removed twice
        assert result.reduction_percentage > 0
        assert result.statistics is not None
        assert result.statistics.original_length == len(result.original_text)
    
    def test_result_operations(self):
        """Test adding operations to result."""
        result = CleaningResult(
            original_text="test text",
            cleaned_text="text"
        )
        
        operation = CleaningOperation(
            operation_type=CleaningOperationType.BOILERPLATE_REMOVAL,
            module_name="TestModule",
            description="Test",
            original_length=9,
            final_length=4
        )
        
        result.add_operation(operation)
        assert len(result.cleaning_operations) == 1
        assert result.statistics.operations_performed == 1
    
    def test_result_warnings_and_entities(self):
        """Test adding warnings and entities."""
        result = CleaningResult(
            original_text="test",
            cleaned_text="test"
        )
        
        result.add_warning("Test warning")
        assert len(result.warnings) == 1
        assert result.statistics.warnings_generated == 1
        
        entity = Entity("test", "TEST", 0, 4, 0.9, EntityType.OTHER)
        result.add_entity(entity)
        assert len(result.entities_preserved) == 1
        assert result.statistics.entities_count == 1
    
    def test_result_summary(self):
        """Test result summary generation."""
        result = CleaningResult(
            original_text="This is test text",
            cleaned_text="This is text",
            quality_score=0.85
        )
        
        summary = result.get_summary()
        assert "Original length: 17" in summary
        assert "Final length: 12" in summary
        assert "Quality score: 0.85" in summary


class TestCleaningModule:
    """Test CleaningModule base class."""
    
    def test_module_creation(self):
        """Test module creation and configuration."""
        config = {"enabled": True, "test_param": "value"}
        module = MockCleaningModule(config)
        
        assert module.is_enabled() is True
        assert module.module_name == "MockCleaningModule"
        assert module.get_config()["test_param"] == "value"
    
    def test_module_enable_disable(self):
        """Test module enable/disable functionality."""
        module = MockCleaningModule()
        
        assert module.is_enabled() is True
        
        module.disable()
        assert module.is_enabled() is False
        
        module.enable()
        assert module.is_enabled() is True
    
    def test_module_processing(self):
        """Test module text processing."""
        module = MockCleaningModule()
        
        text = "This is a test text with test words"
        cleaned_text, operation = module.process_with_timing(text)
        
        assert "test" not in cleaned_text
        assert operation.success is True
        assert operation.items_removed == 2
        assert operation.processing_time >= 0  # Processing time can be 0 for very fast operations
    
    def test_module_disabled_processing(self):
        """Test processing when module is disabled."""
        module = MockCleaningModule()
        module.disable()
        
        text = "This is a test text"
        cleaned_text, operation = module.process_with_timing(text)
        
        assert cleaned_text == text  # No changes when disabled
        assert operation.metadata["disabled"] is True
    
    def test_module_config_update(self):
        """Test module configuration updates."""
        module = MockCleaningModule({"enabled": True})
        
        new_config = {"enabled": False, "new_param": "new_value"}
        module.update_config(new_config)
        
        assert module.is_enabled() is False
        assert module.get_config()["new_param"] == "new_value"


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    def test_default_config_creation(self):
        """Test creation of default configuration."""
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        assert isinstance(config, AdvancedCleaningConfig)
        assert config.enabled is True
        assert len(config.processing_order) > 0
    
    def test_config_file_creation(self):
        """Test configuration file creation and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            
            config_manager = ConfigManager()
            config_manager.create_default_config_file(config_path)
            
            assert config_path.exists()
            
            # Load the created config
            loaded_config = config_manager.load_config(config_path)
            assert loaded_config.enabled is True
    
    def test_config_validation(self):
        """Test configuration validation."""
        config_manager = ConfigManager()
        
        # Valid config
        valid_config = {
            "preprocessing": {
                "advanced_cleaning": {
                    "enabled": True,
                    "modules": {
                        "boilerplate_removal": {
                            "enabled": True
                        }
                    }
                }
            }
        }
        
        errors = config_manager.validate_config(valid_config)
        assert len(errors) == 0
    
    def test_module_config_parsing(self):
        """Test parsing of module-specific configurations."""
        config_dict = {
            "enabled": True,
            "modules": {
                "boilerplate_removal": {
                    "enabled": False,
                    "custom_patterns": ["test_pattern"],
                    "min_line_length": 5
                }
            }
        }
        
        config_manager = ConfigManager()
        config = config_manager._parse_config(config_dict)
        
        assert config.boilerplate_removal.enabled is False
        assert "test_pattern" in config.boilerplate_removal.custom_patterns
        assert config.boilerplate_removal.min_line_length == 5


class TestAdvancedTextCleaner:
    """Test AdvancedTextCleaner orchestrator."""
    
    def test_cleaner_creation(self):
        """Test cleaner creation with default config."""
        cleaner = AdvancedTextCleaner()
        
        assert cleaner.config is not None
        assert isinstance(cleaner.config, AdvancedCleaningConfig)
        assert len(cleaner.processing_order) > 0
    
    def test_module_registration(self):
        """Test module registration and management."""
        cleaner = AdvancedTextCleaner()
        module = MockCleaningModule()
        
        cleaner.register_module("test_module", module)
        
        assert "test_module" in cleaner.get_available_modules()
        assert "test_module" in cleaner.get_enabled_modules()
        
        cleaner.disable_module("test_module")
        assert "test_module" not in cleaner.get_enabled_modules()
        
        cleaner.unregister_module("test_module")
        assert "test_module" not in cleaner.get_available_modules()
    
    def test_text_cleaning_pipeline(self):
        """Test the complete text cleaning pipeline."""
        cleaner = AdvancedTextCleaner()
        
        # Register a mock module
        module = MockCleaningModule()
        cleaner.register_module("boilerplate_removal", module)
        
        text = "This is a test text with test words"
        result = cleaner.clean_text(text)
        
        assert result.success is True
        assert "test" not in result.cleaned_text
        assert len(result.cleaning_operations) == 1
        assert result.statistics is not None
        assert result.processing_time >= 0  # Processing time can be 0 for very fast operations
    
    def test_disabled_cleaning(self):
        """Test behavior when cleaning is disabled."""
        config = AdvancedCleaningConfig(enabled=False)
        cleaner = AdvancedTextCleaner(config)
        
        text = "This is test text"
        result = cleaner.clean_text(text)
        
        assert result.cleaned_text == text
        assert result.metadata["advanced_cleaning_disabled"] is True
    
    def test_file_cleaning(self):
        """Test file-based cleaning."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test file with test content")
            temp_path = f.name
        
        try:
            cleaner = AdvancedTextCleaner()
            module = MockCleaningModule()
            cleaner.register_module("boilerplate_removal", module)
            
            result = cleaner.clean_file(temp_path)
            
            assert result.success is True
            assert "test" not in result.cleaned_text
            assert result.metadata["file_path"] == temp_path
            
        finally:
            Path(temp_path).unlink()
    
    def test_nonexistent_file_cleaning(self):
        """Test cleaning of nonexistent file."""
        cleaner = AdvancedTextCleaner()
        
        result = cleaner.clean_file("nonexistent_file.txt")
        
        assert result.success is False
        assert "File not found" in result.error_message
    
    def test_batch_file_cleaning(self):
        """Test batch file cleaning."""
        # Create temporary files
        temp_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(f"This is test file {i} with test content")
                temp_files.append(f.name)
        
        try:
            cleaner = AdvancedTextCleaner()
            module = MockCleaningModule()
            cleaner.register_module("boilerplate_removal", module)
            
            results = cleaner.clean_files(temp_files)
            
            assert len(results) == 3
            assert all(result.success for result in results)
            assert all("test" not in result.cleaned_text for result in results)
            
        finally:
            for temp_file in temp_files:
                Path(temp_file).unlink()
    
    def test_cleaning_stats(self):
        """Test cleaning statistics generation."""
        cleaner = AdvancedTextCleaner()
        module = MockCleaningModule()
        cleaner.register_module("test_module", module)
        
        stats = cleaner.get_cleaning_stats()
        
        assert "system_info" in stats
        assert "modules" in stats
        assert stats["system_info"]["total_modules"] == 1
        assert "test_module" in stats["modules"]
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        cleaner = AdvancedTextCleaner()
        
        # Default config will have errors since modules aren't registered yet
        errors = cleaner.validate_configuration()
        assert len(errors) > 0  # Expected since no modules are registered
        
        # Register a module and test again
        module = MockCleaningModule()
        cleaner.register_module("boilerplate_removal", module)
        
        # Should have fewer errors now
        new_errors = cleaner.validate_configuration()
        assert len(new_errors) < len(errors)
        
        # Add a module that's in processing order but not registered
        cleaner.processing_order.append("nonexistent_module")
        final_errors = cleaner.validate_configuration()
        assert len(final_errors) > len(new_errors)
        assert any("nonexistent_module" in error for error in final_errors)


if __name__ == "__main__":
    pytest.main([__file__])