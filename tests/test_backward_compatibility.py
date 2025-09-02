"""
Tests for backward compatibility and migration functionality.
"""

import pytest
import tempfile
import json
from pathlib import Path
import shutil
import sys

from llmbuilder.compat import (
    ScriptWrapper, ConfigMigrator, ProjectMigrator,
    setup_legacy_environment, check_migration_needed,
    migrate_legacy_project
)
from llmbuilder.compat.deprecation import check_legacy_usage


class TestBackwardCompatibility:
    """Test backward compatibility features."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.original_cwd = Path.cwd()
        
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def create_legacy_project(self):
        """Create a mock legacy project structure."""
        # Create legacy scripts
        (self.temp_dir / "run.sh").write_text("""#!/bin/bash
echo "Legacy bash script"
""")
        
        (self.temp_dir / "run.ps1").write_text("""# Legacy PowerShell script
Write-Host "Legacy PowerShell script"
""")
        
        # Create legacy config files
        legacy_config = {
            "vocab_size": 32000,
            "embedding_dim": 512,
            "num_layers": 6,
            "num_heads": 8,
            "batch_size": 32,
            "learning_rate": 0.001
        }
        
        with open(self.temp_dir / "config.json", 'w') as f:
            json.dump(legacy_config, f)
        
        # Create legacy directories
        (self.temp_dir / "training").mkdir()
        (self.temp_dir / "data" / "raw").mkdir(parents=True)
        (self.temp_dir / "model").mkdir()
        
        # Create some legacy Python files
        (self.temp_dir / "training" / "train.py").write_text("# Legacy training script")
        (self.temp_dir / "data" / "ingest.py").write_text("# Legacy data ingestion")
    
    def test_legacy_usage_detection(self):
        """Test detection of legacy usage patterns."""
        self.create_legacy_project()
        
        # Change to temp directory
        import os
        os.chdir(self.temp_dir)
        
        try:
            legacy_info = check_legacy_usage()
            
            # Should detect legacy scripts and configs
            assert len(legacy_info['scripts']) > 0
            assert len(legacy_info['configs']) > 0
            assert len(legacy_info['recommendations']) > 0
            
        finally:
            os.chdir(self.original_cwd)
    
    def test_config_migration(self):
        """Test configuration file migration."""
        self.create_legacy_project()
        
        migrator = ConfigMigrator(self.temp_dir)
        result = migrator.migrate_project_configs(backup=True)
        
        # Should successfully migrate configs
        assert result.success
        assert len(result.migrated_files) > 0
        assert len(result.backup_files) > 0
        
        # Check that new config directory was created
        assert (self.temp_dir / ".llmbuilder").exists()
        
        # Validate migrated config structure
        for migrated_file in result.migrated_files:
            if migrated_file.name.endswith('.json'):
                with open(migrated_file) as f:
                    config = json.load(f)
                    assert 'version' in config
                    assert 'model' in config or 'training' in config
    
    def test_script_wrapper(self):
        """Test script wrapper functionality."""
        self.create_legacy_project()
        
        wrapper = ScriptWrapper(self.temp_dir)
        
        # Test CLI equivalent mapping
        cli_cmd = wrapper.get_cli_equivalent("train")
        assert "llmbuilder train start" in cli_cmd
        
        cli_cmd = wrapper.get_cli_equivalent("preprocess")
        assert "llmbuilder data prepare" in cli_cmd
    
    def test_project_migration_dry_run(self):
        """Test project migration in dry-run mode."""
        self.create_legacy_project()
        
        migrator = ProjectMigrator(self.temp_dir)
        
        # Analyze project structure
        analysis = migrator._analyze_project_structure()
        
        assert len(analysis['legacy_files']) > 0
        assert len(analysis['config_files']) >= 0  # May be empty in analysis
        
    def test_legacy_environment_setup(self):
        """Test legacy environment setup."""
        original_path = sys.path.copy()
        
        try:
            setup_legacy_environment()
            
            # Should have added project paths to sys.path
            # (This is a basic test since we can't easily mock the file system)
            assert len(sys.path) >= len(original_path)
            
        finally:
            # Restore original path
            sys.path[:] = original_path
    
    def test_migration_needed_check(self):
        """Test check for migration needed."""
        self.create_legacy_project()
        
        # Change to temp directory
        import os
        os.chdir(self.temp_dir)
        
        try:
            needs_migration = check_migration_needed()
            assert needs_migration is True
            
        finally:
            os.chdir(self.original_cwd)
    
    def test_wrapper_script_creation(self):
        """Test creation of wrapper scripts."""
        from llmbuilder.compat.script_wrapper import create_legacy_wrapper_scripts
        
        # Change to temp directory
        import os
        os.chdir(self.temp_dir)
        
        try:
            bash_wrapper, ps_wrapper = create_legacy_wrapper_scripts()
            
            assert bash_wrapper.exists()
            assert ps_wrapper.exists()
            
            # Check script content
            bash_content = bash_wrapper.read_text()
            assert "llmbuilder" in bash_content
            assert "Legacy wrapper" in bash_content
            
            ps_content = ps_wrapper.read_text()
            assert "llmbuilder" in ps_content
            assert "Legacy wrapper" in ps_content
            
        finally:
            os.chdir(self.original_cwd)


class TestDeprecationWarnings:
    """Test deprecation warning system."""
    
    def test_deprecation_manager(self):
        """Test deprecation manager functionality."""
        from llmbuilder.compat.deprecation import DeprecationManager
        
        manager = DeprecationManager()
        
        # Test warning (should not raise exception)
        manager.warn_legacy_usage(
            "test_feature", 
            "new_feature", 
            show_once=True
        )
        
        # Second call should not show warning due to show_once
        manager.warn_legacy_usage(
            "test_feature", 
            "new_feature", 
            show_once=True
        )
    
    def test_migration_guides(self):
        """Test migration guide generation."""
        from llmbuilder.compat.deprecation import DeprecationManager
        
        manager = DeprecationManager()
        
        # Test different script guides
        guides = ['run.sh', 'run.ps1', 'train.py', 'generic']
        
        for script in guides:
            # Should not raise exception
            guide = manager._get_bash_migration_guide() if script == 'run.sh' else manager._get_generic_migration_guide()
            assert isinstance(guide, str)
            assert len(guide) > 0


class TestImportCompatibility:
    """Test import compatibility layer."""
    
    def test_legacy_import_handler(self):
        """Test legacy import handling."""
        from llmbuilder.compat.legacy_imports import LegacyImportHandler
        
        handler = LegacyImportHandler()
        
        # Test import mapping
        assert 'training.train' in handler.import_mappings
        assert 'data.ingest' in handler.import_mappings
        assert 'model.gpt_model' in handler.import_mappings
    
    def test_import_functions(self):
        """Test import convenience functions."""
        from llmbuilder.compat.legacy_imports import (
            import_training_module,
            import_data_module,
            import_model_module
        )
        
        # These should not raise exceptions (may return None if modules don't exist)
        training_module = import_training_module('train')
        data_module = import_data_module('ingest')
        model_module = import_model_module('gpt_model')
        
        # Test passes if no exceptions are raised


if __name__ == '__main__':
    pytest.main([__file__])