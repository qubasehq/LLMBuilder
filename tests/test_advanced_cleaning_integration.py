"""
Integration tests for the Advanced Cybersecurity Dataset Cleaning system.

This module tests the integration of the advanced cleaning system with the
existing configuration and preprocessing pipeline.
"""

import pytest
import json
from pathlib import Path
from llmbuilder.core.training.advanced_cleaning import AdvancedTextCleaner, ConfigManager


class TestAdvancedCleaningIntegration:
    """Test integration with existing system."""
    
    def test_config_loading_from_main_config(self):
        """Test loading advanced cleaning config from main config.json."""
        config_path = Path("config.json")
        
        if not config_path.exists():
            pytest.skip("Main config.json not found")
        
        # Load the main config
        with open(config_path, 'r', encoding='utf-8') as f:
            main_config = json.load(f)
        
        # Check if advanced cleaning config exists
        advanced_config = main_config.get('preprocessing', {}).get('advanced_cleaning')
        
        if not advanced_config:
            pytest.skip("Advanced cleaning config not found in main config")
        
        # Test that we can create a ConfigManager and load the config
        config_manager = ConfigManager()
        parsed_config = config_manager._parse_config(advanced_config)
        
        assert parsed_config.enabled is True
        assert len(parsed_config.processing_order) > 0
        assert parsed_config.boilerplate_removal.enabled is True
        assert len(parsed_config.boilerplate_removal.custom_patterns) > 0
    
    def test_advanced_cleaner_with_main_config(self):
        """Test creating AdvancedTextCleaner with main config."""
        config_path = Path("config.json")
        
        if not config_path.exists():
            pytest.skip("Main config.json not found")
        
        # Load the main config
        with open(config_path, 'r', encoding='utf-8') as f:
            main_config = json.load(f)
        
        advanced_config = main_config.get('preprocessing', {}).get('advanced_cleaning')
        
        if not advanced_config:
            pytest.skip("Advanced cleaning config not found in main config")
        
        # Create cleaner with the config
        cleaner = AdvancedTextCleaner(advanced_config)
        
        assert cleaner.config.enabled is True
        assert len(cleaner.processing_order) > 0
        
        # Test basic functionality (without actual modules registered)
        stats = cleaner.get_cleaning_stats()
        assert "system_info" in stats
        assert stats["system_info"]["enabled"] is True
    
    def test_config_validation_with_main_config(self):
        """Test configuration validation with main config."""
        config_path = Path("config.json")
        
        if not config_path.exists():
            pytest.skip("Main config.json not found")
        
        # Load and validate the main config
        with open(config_path, 'r', encoding='utf-8') as f:
            main_config = json.load(f)
        
        config_manager = ConfigManager()
        errors = config_manager.validate_config(main_config)
        
        # Should have no validation errors
        assert len(errors) == 0, f"Configuration validation errors: {errors}"
    
    def test_cybersecurity_keywords_present(self):
        """Test that cybersecurity-specific keywords are configured."""
        config_path = Path("config.json")
        
        if not config_path.exists():
            pytest.skip("Main config.json not found")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            main_config = json.load(f)
        
        advanced_config = main_config.get('preprocessing', {}).get('advanced_cleaning')
        
        if not advanced_config:
            pytest.skip("Advanced cleaning config not found in main config")
        
        domain_config = advanced_config.get('modules', {}).get('domain_filtering', {})
        keywords = domain_config.get('cybersecurity_keywords', [])
        
        # Check for key cybersecurity terms
        expected_keywords = ['firewall', 'malware', 'vulnerability', 'encryption', 'security']
        for keyword in expected_keywords:
            assert keyword in keywords, f"Missing cybersecurity keyword: {keyword}"
    
    def test_entity_types_configured(self):
        """Test that cybersecurity entity types are configured."""
        config_path = Path("config.json")
        
        if not config_path.exists():
            pytest.skip("Main config.json not found")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            main_config = json.load(f)
        
        advanced_config = main_config.get('preprocessing', {}).get('advanced_cleaning')
        
        if not advanced_config:
            pytest.skip("Advanced cleaning config not found in main config")
        
        entity_config = advanced_config.get('modules', {}).get('entity_preservation', {})
        entity_types = entity_config.get('entity_types', [])
        
        # Check for key cybersecurity entity types
        expected_types = ['CVE', 'PROTOCOL', 'TOOL', 'VULNERABILITY']
        for entity_type in expected_types:
            assert entity_type in entity_types, f"Missing entity type: {entity_type}"


if __name__ == "__main__":
    pytest.main([__file__])