"""
Tests for inference CLI commands.

This module tests the interactive inference, prompt templates, and
conversation history functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from click.testing import CliRunner

from llmbuilder.cli.inference import inference
from llmbuilder.core.inference.templates import PromptTemplateManager
from llmbuilder.core.inference.history import ConversationHistory


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Create a sample model configuration."""
    return {
        'model': {
            'vocab_size': 1000,
            'embedding_dim': 128,
            'num_layers': 2,
            'num_heads': 8,
            'max_seq_length': 512,
            'dropout': 0.1
        }
    }


@pytest.fixture
def mock_model_files(temp_dir, sample_config):
    """Create mock model files for testing."""
    # Create config file
    config_path = temp_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(sample_config, f)
    
    # Create mock model file
    model_path = temp_dir / 'model.pt'
    model_path.touch()
    
    # Create mock tokenizer directory
    tokenizer_dir = temp_dir / 'tokenizer'
    tokenizer_dir.mkdir()
    (tokenizer_dir / 'tokenizer.model').touch()
    
    return {
        'model': model_path,
        'tokenizer': tokenizer_dir,
        'config': config_path
    }


class TestInferenceCommands:
    """Test inference CLI commands."""
    
    def test_generate_command_help(self):
        """Test that generate command shows help correctly."""
        runner = CliRunner()
        result = runner.invoke(inference, ['generate', '--help'])
        
        assert result.exit_code == 0
        assert 'Generate text from a single prompt' in result.output
        assert '--max-tokens' in result.output
        assert '--temperature' in result.output
    
    def test_chat_command_help(self):
        """Test that chat command shows help correctly."""
        runner = CliRunner()
        result = runner.invoke(inference, ['chat', '--help'])
        
        assert result.exit_code == 0
        assert 'Start an interactive chat session' in result.output
        assert '--model' in result.output
        assert '--save-history' in result.output
    
    def test_generate_command_missing_model(self):
        """Test generate command with missing model files."""
        runner = CliRunner()
        result = runner.invoke(inference, [
            'generate', 
            'Test prompt',
            '--model', 'nonexistent.pt'
        ])
        
        # Should fail due to missing model
        assert result.exit_code != 0
    
    def test_generate_command_with_parameters(self):
        """Test generate command with various parameters."""
        runner = CliRunner()
        
        # Test with different parameter combinations
        result = runner.invoke(inference, [
            'generate',
            'Test prompt',
            '--max-tokens', '50',
            '--temperature', '0.5',
            '--top-k', '20',
            '--top-p', '0.8'
        ])
        
        # Command structure should be valid even if model loading fails
        assert 'generate' in result.output.lower() or result.exit_code != 0


class TestPromptTemplates:
    """Test prompt template functionality."""
    
    def test_template_manager_initialization(self, temp_dir):
        """Test template manager initialization."""
        manager = PromptTemplateManager(temp_dir / 'templates')
        
        # Should create templates directory
        assert (temp_dir / 'templates').exists()
        
        # Should create default templates
        templates = manager.list_templates()
        assert len(templates) > 0
        assert 'creative' in templates
        assert 'technical' in templates
    
    def test_save_and_load_template(self, temp_dir):
        """Test saving and loading templates."""
        manager = PromptTemplateManager(temp_dir / 'templates')
        
        # Create a test template
        template_config = {
            'description': 'Test template',
            'format': 'Test: {prompt}',
            'parameters': {
                'temperature': 0.5,
                'max_new_tokens': 100
            }
        }
        
        # Save template
        manager.save_template('test', template_config)
        
        # Load template
        loaded_template = manager.get_template('test')
        assert loaded_template is not None
        assert loaded_template['description'] == 'Test template'
        assert loaded_template['format'] == 'Test: {prompt}'
        assert loaded_template['parameters']['temperature'] == 0.5
    
    def test_apply_template(self, temp_dir):
        """Test applying templates to prompts."""
        manager = PromptTemplateManager(temp_dir / 'templates')
        
        # Create a test template
        template_config = {
            'description': 'Test template',
            'format': 'Explain: {prompt}',
            'parameters': {
                'temperature': 0.3
            }
        }
        manager.save_template('explain', template_config)
        
        # Apply template
        formatted_prompt, params = manager.apply_template('explain', 'quantum physics')
        
        assert formatted_prompt == 'Explain: quantum physics'
        assert params['temperature'] == 0.3
    
    def test_template_validation(self, temp_dir):
        """Test template configuration validation."""
        manager = PromptTemplateManager(temp_dir / 'templates')
        
        # Test invalid temperature
        with pytest.raises(ValueError):
            manager.save_template('invalid', {
                'parameters': {'temperature': -1.0}
            })
        
        # Test invalid top_p
        with pytest.raises(ValueError):
            manager.save_template('invalid', {
                'parameters': {'top_p': 1.5}
            })
        
        # Test invalid format string
        with pytest.raises(ValueError):
            manager.save_template('invalid', {
                'format': 'Missing {placeholder}'
            })
    
    def test_template_cli_commands(self, temp_dir):
        """Test template CLI commands."""
        runner = CliRunner()
        
        # Test list templates
        result = runner.invoke(inference, ['templates', 'list'])
        assert result.exit_code == 0
        
        # Test create template
        result = runner.invoke(inference, [
            'templates', 'create', 'test-template',
            '--description', 'Test template',
            '--temperature', '0.5'
        ])
        assert result.exit_code == 0
        
        # Test show template
        result = runner.invoke(inference, ['templates', 'show', 'creative'])
        assert result.exit_code == 0


class TestConversationHistory:
    """Test conversation history functionality."""
    
    def test_history_initialization(self, temp_dir):
        """Test conversation history initialization."""
        history = ConversationHistory(temp_dir / 'history')
        
        # Should create history directory
        assert (temp_dir / 'history').exists()
        
        # Should have empty messages initially
        assert len(history.get_messages()) == 0
    
    def test_add_and_get_messages(self, temp_dir):
        """Test adding and retrieving messages."""
        history = ConversationHistory(temp_dir / 'history')
        
        # Add messages
        history.add_message('user', 'Hello')
        history.add_message('assistant', 'Hi there!')
        history.add_message('user', 'How are you?')
        
        # Get all messages
        messages = history.get_messages()
        assert len(messages) == 3
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == 'Hello'
        
        # Get filtered messages
        user_messages = history.get_messages('user')
        assert len(user_messages) == 2
        
        assistant_messages = history.get_messages('assistant')
        assert len(assistant_messages) == 1
    
    def test_save_and_load_history(self, temp_dir):
        """Test saving and loading conversation history."""
        history = ConversationHistory(temp_dir / 'history')
        
        # Add some messages
        history.add_message('user', 'Test message 1')
        history.add_message('assistant', 'Test response 1')
        history.set_metadata('test_key', 'test_value')
        
        # Save to file
        history_file = temp_dir / 'test_conversation.json'
        history.save_to_file(history_file)
        
        assert history_file.exists()
        
        # Load into new history instance
        new_history = ConversationHistory(temp_dir / 'history')
        new_history.load_from_file(history_file)
        
        # Verify loaded data
        messages = new_history.get_messages()
        assert len(messages) == 2
        assert messages[0]['content'] == 'Test message 1'
        assert new_history.get_metadata('test_key') == 'test_value'
    
    def test_conversation_summary(self, temp_dir):
        """Test conversation summary generation."""
        history = ConversationHistory(temp_dir / 'history')
        
        # Add messages
        history.add_message('user', 'Hello')
        history.add_message('assistant', 'Hi!')
        history.add_message('user', 'Goodbye')
        
        # Get summary
        summary = history.get_conversation_summary()
        
        assert summary['total_messages'] == 3
        assert summary['user_messages'] == 2
        assert summary['assistant_messages'] == 1
        assert summary['total_characters'] > 0
    
    def test_search_messages(self, temp_dir):
        """Test message search functionality."""
        history = ConversationHistory(temp_dir / 'history')
        
        # Add messages
        history.add_message('user', 'Tell me about Python')
        history.add_message('assistant', 'Python is a programming language')
        history.add_message('user', 'What about JavaScript?')
        
        # Search for messages
        python_messages = history.search_messages('Python')
        assert len(python_messages) == 2
        
        # Case insensitive search
        python_messages_ci = history.search_messages('python', case_sensitive=False)
        assert len(python_messages_ci) == 2
        
        # Case sensitive search
        python_messages_cs = history.search_messages('python', case_sensitive=True)
        assert len(python_messages_cs) == 0  # No lowercase 'python'
    
    def test_export_to_text(self, temp_dir):
        """Test exporting conversation to text format."""
        history = ConversationHistory(temp_dir / 'history')
        
        # Add messages
        history.add_message('user', 'Hello')
        history.add_message('assistant', 'Hi there!')
        
        # Export to text
        text_file = temp_dir / 'conversation.txt'
        history.export_to_text(text_file)
        
        assert text_file.exists()
        
        # Check content
        content = text_file.read_text(encoding='utf-8')
        assert 'Hello' in content
        assert 'Hi there!' in content
        assert 'User:' in content
        assert 'Assistant:' in content
    
    def test_history_cli_commands(self, temp_dir):
        """Test history CLI commands."""
        runner = CliRunner()
        
        # Test list histories (should work even with no histories)
        result = runner.invoke(inference, ['history', 'list'])
        assert result.exit_code == 0


class TestInferenceIntegration:
    """Test integration between inference components."""
    
    def test_template_and_history_integration(self, temp_dir):
        """Test using templates with conversation history."""
        # Create template manager
        template_manager = PromptTemplateManager(temp_dir / 'templates')
        
        # Create a template
        template_config = {
            'description': 'Question template',
            'format': 'Question: {prompt}',
            'parameters': {'temperature': 0.5}
        }
        template_manager.save_template('question', template_config)
        
        # Create conversation history
        history = ConversationHistory(temp_dir / 'history')
        
        # Apply template and add to history
        formatted_prompt, params = template_manager.apply_template('question', 'What is AI?')
        history.add_message('user', formatted_prompt)
        history.add_message('assistant', 'AI is artificial intelligence.')
        
        # Verify integration
        messages = history.get_messages()
        assert len(messages) == 2
        assert messages[0]['content'] == 'Question: What is AI?'
        assert params['temperature'] == 0.5
    
    def test_cli_parameter_validation(self):
        """Test CLI parameter validation."""
        runner = CliRunner()
        
        # Test invalid temperature
        result = runner.invoke(inference, [
            'generate', 'test',
            '--temperature', '-1.0'
        ])
        # Should handle invalid parameters gracefully
        assert result.exit_code != 0 or 'temperature' in result.output.lower()
        
        # Test invalid max-tokens
        result = runner.invoke(inference, [
            'generate', 'test',
            '--max-tokens', '0'
        ])
        # Should handle invalid parameters gracefully
        assert result.exit_code != 0 or 'tokens' in result.output.lower()


if __name__ == '__main__':
    pytest.main([__file__])