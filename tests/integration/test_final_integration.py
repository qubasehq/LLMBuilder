"""
Final integration tests for the complete LLMBuilder CLI system.
"""

import pytest
import tempfile
import json
import shutil
from pathlib import Path
from click.testing import CliRunner

from llmbuilder.cli.main import cli
from llmbuilder.utils.config import ConfigManager
from llmbuilder.utils.workflow import WorkflowManager


class TestFinalIntegration:
    """Test complete workflows and cross-command integration."""
    
    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()
    
    def test_complete_project_lifecycle(self, runner, temp_workspace):
        """Test complete project lifecycle from init to deployment."""
        
        # Change to project directory
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_workspace)
            
            # 1. Initialize project
            result = runner.invoke(cli, ['init', 'test-project', '--template', 'default'])
            assert result.exit_code == 0
            assert "Project 'test-project' created successfully" in result.output
            
            # Verify project structure
            project_dir = temp_workspace / "test-project"
            assert project_dir.exists()
            assert (project_dir / ".llmbuilder").exists()
            assert (project_dir / "data").exists()
            assert (project_dir / "models").exists()
            
            # Change to project directory
            os.chdir(project_dir)
            
            # 2. Configure project
            result = runner.invoke(cli, ['config', 'set', 'training.epochs', '2'])
            assert result.exit_code == 0
            
            result = runner.invoke(cli, ['config', 'get', 'training.epochs'])
            assert result.exit_code == 0
            assert "2" in result.output
            
            # 3. Create sample data
            data_dir = project_dir / "data" / "raw"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            sample_data = data_dir / "sample.txt"
            sample_data.write_text("This is sample training data for testing.\n" * 100)
            
            # 4. Prepare data
            result = runner.invoke(cli, [
                'data', 'prepare',
                '--input', str(data_dir),
                '--output', str(project_dir / "data" / "processed")
            ])
            assert result.exit_code == 0
            
            # 5. Split data
            result = runner.invoke(cli, [
                'data', 'split',
                '--input', str(project_dir / "data" / "processed"),
                '--ratios', '0.8,0.1,0.1'
            ])
            assert result.exit_code == 0
            
            # 6. List available models (should work without errors)
            result = runner.invoke(cli, ['model', 'list', '--source', 'local'])
            assert result.exit_code == 0
            
            # 7. Test configuration management
            result = runner.invoke(cli, ['config', 'list'])
            assert result.exit_code == 0
            assert "training" in result.output
            
            # 8. Test help system
            result = runner.invoke(cli, ['help'])
            assert result.exit_code == 0
            
            result = runner.invoke(cli, ['--help'])
            assert result.exit_code == 0
            
        finally:
            os.chdir(original_cwd)
    
    def test_pipeline_creation_and_management(self, runner, temp_workspace):
        """Test pipeline creation and workflow management."""
        
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_workspace)
            
            # Initialize project first
            runner.invoke(cli, ['init', 'pipeline-test'])
            os.chdir(temp_workspace / "pipeline-test")
            
            # Create sample data
            data_dir = temp_workspace / "pipeline-test" / "data" / "raw"
            data_dir.mkdir(parents=True, exist_ok=True)
            (data_dir / "sample.txt").write_text("Sample data for pipeline testing.\n" * 50)
            
            # Create training pipeline (dry run)
            result = runner.invoke(cli, [
                'pipeline', 'train', 'test-pipeline',
                '--data-path', str(data_dir),
                '--model', 'gpt2',
                '--output-dir', './output',
                '--dry-run'
            ])
            assert result.exit_code == 0
            assert "Pipeline 'test-pipeline' would execute" in result.output
            
            # List pipelines
            result = runner.invoke(cli, ['pipeline', 'list'])
            assert result.exit_code == 0
            
        finally:
            os.chdir(original_cwd)
    
    def test_cross_command_data_sharing(self, runner, temp_workspace):
        """Test data sharing between commands."""
        
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_workspace)
            
            # Initialize project
            runner.invoke(cli, ['init', 'data-sharing-test'])
            os.chdir(temp_workspace / "data-sharing-test")
            
            # Set configuration that should be shared
            result = runner.invoke(cli, ['config', 'set', 'model.max_length', '512'])
            assert result.exit_code == 0
            
            # Verify configuration is accessible from different commands
            result = runner.invoke(cli, ['config', 'get', 'model.max_length'])
            assert result.exit_code == 0
            assert "512" in result.output
            
            # Test that configuration persists
            config_manager = ConfigManager()
            config = config_manager.load_config()
            assert config['model']['max_length'] == 512
            
        finally:
            os.chdir(original_cwd)
    
    def test_error_handling_and_recovery(self, runner, temp_workspace):
        """Test error handling and recovery mechanisms."""
        
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_workspace)
            
            # Test invalid command
            result = runner.invoke(cli, ['invalid-command'])
            assert result.exit_code != 0
            assert "No such command" in result.output
            
            # Test invalid configuration
            result = runner.invoke(cli, ['config', 'set', 'invalid.key', 'value'])
            # Should handle gracefully (may warn but not crash)
            
            # Test missing project initialization
            result = runner.invoke(cli, ['config', 'get', 'training.epochs'])
            # Should provide helpful error message
            
        finally:
            os.chdir(original_cwd)
    
    def test_workflow_manager_integration(self, temp_workspace):
        """Test workflow manager functionality."""
        
        workflow_manager = WorkflowManager(temp_workspace)
        
        # Create a test workflow
        steps = [
            {"command": "data prepare", "args": {"input": "./data", "output": "./processed"}},
            {"command": "train start", "args": {"epochs": 1, "batch_size": 2}}
        ]
        
        workflow_id = workflow_manager.create_workflow("test-workflow", steps)
        assert workflow_id is not None
        
        # Load workflow
        workflow_data = workflow_manager.load_workflow(workflow_id)
        assert workflow_data["name"] == "test-workflow"
        assert len(workflow_data["steps"]) == 2
        
        # Test shared data
        workflow_manager.set_shared_data(workflow_id, "test_key", "test_value")
        shared_data = workflow_manager.get_shared_data(workflow_id, "test_key")
        assert shared_data == "test_value"
        
        # List workflows
        workflows = workflow_manager.list_workflows()
        assert len(workflows) >= 1
        assert any(w["id"] == workflow_id for w in workflows)
    
    def test_command_chaining(self, runner, temp_workspace):
        """Test command chaining and pipeline execution."""
        
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_workspace)
            
            # Initialize project
            runner.invoke(cli, ['init', 'chaining-test'])
            os.chdir(temp_workspace / "chaining-test")
            
            # Test that commands can be chained through configuration
            result = runner.invoke(cli, ['config', 'set', 'training.batch_size', '4'])
            assert result.exit_code == 0
            
            # Verify the setting is available for other commands
            result = runner.invoke(cli, ['config', 'get', 'training.batch_size'])
            assert result.exit_code == 0
            assert "4" in result.output
            
            # Test help system integration
            result = runner.invoke(cli, ['help', 'train'])
            assert result.exit_code == 0
            
        finally:
            os.chdir(original_cwd)
    
    def test_comprehensive_help_system(self, runner):
        """Test the comprehensive help and documentation system."""
        
        # Test main help
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert "LLMBuilder - Complete LLM Training and Deployment Pipeline" in result.output
        
        # Test command-specific help
        commands = ['init', 'config', 'data', 'model', 'train', 'eval', 'deploy']
        for command in commands:
            result = runner.invoke(cli, [command, '--help'])
            assert result.exit_code == 0, f"Help for {command} failed"
        
        # Test interactive help
        result = runner.invoke(cli, ['help'])
        assert result.exit_code == 0
        
        # Test examples
        result = runner.invoke(cli, ['examples'])
        assert result.exit_code == 0
        
        # Test command discovery
        result = runner.invoke(cli, ['discover'])
        assert result.exit_code == 0
    
    def test_configuration_hierarchy(self, runner, temp_workspace):
        """Test configuration hierarchy and precedence."""
        
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_workspace)
            
            # Initialize project
            runner.invoke(cli, ['init', 'config-test'])
            os.chdir(temp_workspace / "config-test")
            
            # Set project-level configuration
            result = runner.invoke(cli, ['config', 'set', 'training.learning_rate', '0.001'])
            assert result.exit_code == 0
            
            # Verify configuration is set
            result = runner.invoke(cli, ['config', 'get', 'training.learning_rate'])
            assert result.exit_code == 0
            assert "0.001" in result.output
            
            # Test configuration with command-line override
            result = runner.invoke(cli, [
                '--config', str(temp_workspace / "config-test" / ".llmbuilder" / "config.json"),
                'config', 'get', 'training.learning_rate'
            ])
            assert result.exit_code == 0
            
        finally:
            os.chdir(original_cwd)
    
    def test_logging_and_monitoring_integration(self, runner, temp_workspace):
        """Test logging and monitoring system integration."""
        
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_workspace)
            
            # Initialize project
            runner.invoke(cli, ['init', 'monitoring-test'])
            os.chdir(temp_workspace / "monitoring-test")
            
            # Test verbose logging
            result = runner.invoke(cli, ['--verbose', 'config', 'list'])
            assert result.exit_code == 0
            
            # Test quiet mode
            result = runner.invoke(cli, ['--quiet', 'config', 'list'])
            assert result.exit_code == 0
            
            # Test monitor commands
            result = runner.invoke(cli, ['monitor', '--help'])
            assert result.exit_code == 0
            
        finally:
            os.chdir(original_cwd)
    
    def test_upgrade_and_maintenance(self, runner):
        """Test upgrade and maintenance functionality."""
        
        # Test upgrade check
        result = runner.invoke(cli, ['upgrade', '--check'])
        assert result.exit_code == 0
        
        # Test version display
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
    
    def test_backward_compatibility(self, runner, temp_project):
        """Test backward compatibility features."""
        
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_project)
            
            # Test migration command
            result = runner.invoke(cli, ['migrate', '--help'])
            assert result.exit_code == 0
            
        finally:
            os.chdir(original_cwd)


class TestPerformanceAndScalability:
    """Test performance and scalability aspects."""
    
    def test_large_configuration_handling(self, temp_workspace):
        """Test handling of large configuration files."""
        
        config_manager = ConfigManager()
        
        # Create large configuration
        large_config = {
            "model": {f"param_{i}": f"value_{i}" for i in range(1000)},
            "training": {f"setting_{i}": i for i in range(1000)},
            "data": {f"option_{i}": f"data_{i}" for i in range(1000)}
        }
        
        config_file = temp_workspace / "large_config.json"
        with open(config_file, 'w') as f:
            json.dump(large_config, f)
        
        # Test loading large configuration
        loaded_config = config_manager.load_config(config_file)
        assert len(loaded_config["model"]) == 1000
        assert len(loaded_config["training"]) == 1000
        assert len(loaded_config["data"]) == 1000
    
    def test_workflow_scalability(self, temp_workspace):
        """Test workflow manager with many workflows."""
        
        workflow_manager = WorkflowManager(temp_workspace)
        
        # Create multiple workflows
        workflow_ids = []
        for i in range(10):
            steps = [
                {"command": f"step_{j}", "args": {"param": f"value_{j}"}}
                for j in range(5)
            ]
            workflow_id = workflow_manager.create_workflow(f"workflow_{i}", steps)
            workflow_ids.append(workflow_id)
        
        # Test listing many workflows
        workflows = workflow_manager.list_workflows()
        assert len(workflows) >= 10
        
        # Test loading each workflow
        for workflow_id in workflow_ids:
            workflow_data = workflow_manager.load_workflow(workflow_id)
            assert workflow_data is not None
            assert len(workflow_data["steps"]) == 5


class TestErrorHandlingAndRecovery:
    """Test comprehensive error handling and recovery."""
    
    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()
    
    def test_graceful_error_handling(self, runner):
        """Test graceful handling of various error conditions."""
        
        # Test invalid arguments
        result = runner.invoke(cli, ['config', 'set'])  # Missing arguments
        assert result.exit_code != 0
        
        # Test non-existent file
        result = runner.invoke(cli, ['--config', '/non/existent/file.json', 'config', 'list'])
        assert result.exit_code != 0
        
        # Test invalid JSON in config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            f.flush()
            
            result = runner.invoke(cli, ['--config', f.name, 'config', 'list'])
            assert result.exit_code != 0
    
    def test_recovery_mechanisms(self, runner, temp_workspace):
        """Test recovery mechanisms for common issues."""
        
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_workspace)
            
            # Initialize project
            runner.invoke(cli, ['init', 'recovery-test'])
            os.chdir(temp_workspace / "recovery-test")
            
            # Test configuration reset
            result = runner.invoke(cli, ['config', 'reset'])
            assert result.exit_code == 0
            
            # Test that configuration is restored to defaults
            result = runner.invoke(cli, ['config', 'get', 'training.epochs'])
            assert result.exit_code == 0
            
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])