#!/usr/bin/env python3
"""
Test data fixtures for LLMBuilder tests.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import tempfile
import shutil


class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def create_sample_jsonl_dataset(size: int = 100) -> List[Dict[str, Any]]:
        """Create sample JSONL dataset."""
        dataset = []
        
        for i in range(size):
            item = {
                "text": f"This is sample text number {i}. It contains various content for testing purposes.",
                "id": i,
                "category": "positive" if i % 2 == 0 else "negative",
                "length": len(f"This is sample text number {i}. It contains various content for testing purposes."),
                "metadata": {
                    "source": "test_generator",
                    "created_at": f"2024-01-{(i % 30) + 1:02d}",
                    "quality_score": 0.8 + (i % 3) * 0.1
                }
            }
            dataset.append(item)
        
        return dataset
    
    @staticmethod
    def create_conversation_dataset(size: int = 50) -> List[Dict[str, Any]]:
        """Create conversation-style dataset."""
        dataset = []
        
        conversation_templates = [
            {
                "input": "What is the capital of France?",
                "output": "The capital of France is Paris."
            },
            {
                "input": "How do you make coffee?",
                "output": "To make coffee, you need coffee beans, hot water, and a brewing method like a coffee maker or French press."
            },
            {
                "input": "Explain machine learning in simple terms.",
                "output": "Machine learning is a way for computers to learn patterns from data without being explicitly programmed for each task."
            }
        ]
        
        for i in range(size):
            template = conversation_templates[i % len(conversation_templates)]
            item = {
                "input": template["input"].replace("France", f"Country{i}").replace("coffee", f"beverage{i}"),
                "output": template["output"].replace("France", f"Country{i}").replace("coffee", f"beverage{i}"),
                "conversation_id": f"conv_{i // 10}",
                "turn": i % 10,
                "quality": "high" if i % 3 == 0 else "medium"
            }
            dataset.append(item)
        
        return dataset
    
    @staticmethod
    def create_multilingual_dataset(size: int = 30) -> List[Dict[str, Any]]:
        """Create multilingual dataset."""
        dataset = []
        
        texts = {
            "en": "Hello, how are you today?",
            "es": "Hola, ¿cómo estás hoy?",
            "fr": "Bonjour, comment allez-vous aujourd'hui?",
            "de": "Hallo, wie geht es dir heute?",
            "it": "Ciao, come stai oggi?",
            "pt": "Olá, como você está hoje?"
        }
        
        languages = list(texts.keys())
        
        for i in range(size):
            lang = languages[i % len(languages)]
            item = {
                "text": texts[lang] + f" (Sample {i})",
                "language": lang,
                "id": i,
                "translation_available": True,
                "script": "latin"
            }
            dataset.append(item)
        
        return dataset
    
    @staticmethod
    def create_code_dataset(size: int = 20) -> List[Dict[str, Any]]:
        """Create code-related dataset."""
        dataset = []
        
        code_examples = [
            {
                "language": "python",
                "code": "def hello_world():\n    print('Hello, World!')\n    return True",
                "description": "A simple hello world function"
            },
            {
                "language": "javascript",
                "code": "function addNumbers(a, b) {\n    return a + b;\n}",
                "description": "Function to add two numbers"
            },
            {
                "language": "java",
                "code": "public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n    }\n}",
                "description": "Java hello world program"
            }
        ]
        
        for i in range(size):
            example = code_examples[i % len(code_examples)]
            item = {
                "code": example["code"].replace("Hello", f"Hello{i}"),
                "language": example["language"],
                "description": example["description"] + f" (Example {i})",
                "complexity": "beginner" if i % 3 == 0 else "intermediate",
                "tags": ["example", "tutorial", example["language"]],
                "id": i
            }
            dataset.append(item)
        
        return dataset
    
    @staticmethod
    def create_noisy_dataset(size: int = 50) -> List[Dict[str, Any]]:
        """Create dataset with various types of noise."""
        dataset = []
        
        for i in range(size):
            # Add different types of noise
            if i % 5 == 0:
                # Extra whitespace
                text = f"   This is text with extra whitespace {i}.   \n\n\n"
            elif i % 5 == 1:
                # HTML tags
                text = f"<p>This is text with <b>HTML tags</b> number {i}.</p>"
            elif i % 5 == 2:
                # URLs and emails
                text = f"This text {i} contains http://example.com and test@email.com"
            elif i % 5 == 3:
                # Special characters
                text = f"Text with special chars: @#$%^&*() number {i}!!!"
            else:
                # Normal text
                text = f"This is clean text number {i}."
            
            item = {
                "text": text,
                "id": i,
                "noise_type": ["whitespace", "html", "urls", "special_chars", "clean"][i % 5],
                "needs_cleaning": i % 5 != 4
            }
            dataset.append(item)
        
        return dataset


class TestFileManager:
    """Manage test files and directories."""
    
    def __init__(self):
        self.temp_dirs = []
    
    def create_temp_directory(self) -> Path:
        """Create temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def create_test_project_structure(self, base_path: Path) -> Dict[str, Path]:
        """Create standard test project structure."""
        directories = {
            'data_raw': base_path / 'data' / 'raw',
            'data_cleaned': base_path / 'data' / 'cleaned',
            'data_deduped': base_path / 'data' / 'deduped',
            'data_tokens': base_path / 'data' / 'tokens',
            'data_finetune': base_path / 'data' / 'finetune',
            'exports_checkpoints': base_path / 'exports' / 'checkpoints',
            'exports_gguf': base_path / 'exports' / 'gguf',
            'exports_tokenizer': base_path / 'exports' / 'tokenizer',
            'logs': base_path / 'logs',
            'models': base_path / 'models',
            'configs': base_path / 'configs'
        }
        
        for dir_path in directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        return directories
    
    def write_jsonl_file(self, file_path: Path, data: List[Dict[str, Any]]):
        """Write data to JSONL file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def write_json_file(self, file_path: Path, data: Dict[str, Any]):
        """Write data to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def write_text_file(self, file_path: Path, content: str):
        """Write content to text file."""
        file_path.write_text(content, encoding='utf-8')
    
    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        self.temp_dirs.clear()


class MockConfigData:
    """Mock configuration data for testing."""
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration for testing."""
        return {
            "model": {
                "architecture": "gpt",
                "vocab_size": 50000,
                "embedding_dim": 768,
                "num_layers": 12,
                "num_heads": 12,
                "hidden_dim": 3072,
                "max_seq_length": 1024,
                "dropout": 0.1
            },
            "training": {
                "batch_size": 8,
                "learning_rate": 5e-5,
                "num_epochs": 3,
                "max_steps": 1000,
                "warmup_steps": 100,
                "save_every": 100,
                "eval_every": 50,
                "log_every": 10,
                "optimizer": "adamw",
                "scheduler": "linear",
                "mixed_precision": True,
                "gradient_checkpointing": False,
                "gradient_accumulation_steps": 1
            },
            "data": {
                "input_formats": ["txt", "json", "jsonl", "csv"],
                "max_length": 512,
                "min_length": 10,
                "preprocessing": {
                    "normalize_whitespace": True,
                    "remove_html": True,
                    "remove_urls": False,
                    "lowercase": False
                },
                "deduplication": {
                    "enabled": True,
                    "method": "hash",
                    "threshold": 0.9
                }
            },
            "tokenizer": {
                "type": "bpe",
                "vocab_size": 50000,
                "special_tokens": ["<pad>", "<unk>", "<s>", "</s>"],
                "model_type": "unigram"
            },
            "deployment": {
                "api_framework": "fastapi",
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 1,
                "timeout": 30,
                "cors_enabled": True
            },
            "monitoring": {
                "log_level": "INFO",
                "metrics_enabled": True,
                "dashboard_port": 8080,
                "export_format": "json"
            }
        }
    
    @staticmethod
    def get_minimal_config() -> Dict[str, Any]:
        """Get minimal configuration for testing."""
        return {
            "model": {
                "vocab_size": 1000,
                "embedding_dim": 64,
                "num_layers": 2,
                "num_heads": 2,
                "max_seq_length": 128
            },
            "training": {
                "batch_size": 2,
                "learning_rate": 0.001,
                "num_epochs": 1,
                "max_steps": 10
            }
        }
    
    @staticmethod
    def get_invalid_config() -> Dict[str, Any]:
        """Get invalid configuration for testing error handling."""
        return {
            "model": {
                "vocab_size": "invalid",  # Should be int
                "embedding_dim": -1,      # Should be positive
                "num_layers": 0           # Should be positive
            },
            "training": {
                "batch_size": 0,          # Should be positive
                "learning_rate": "high",  # Should be float
                "num_epochs": -1          # Should be positive
            }
        }


class MockModelData:
    """Mock model data for testing."""
    
    @staticmethod
    def get_model_info_list() -> List[Dict[str, Any]]:
        """Get list of mock model information."""
        return [
            {
                "name": "gpt2",
                "size": "124M",
                "description": "GPT-2 small model",
                "architecture": "gpt2",
                "vocab_size": 50257,
                "max_length": 1024,
                "available": True,
                "source": "huggingface"
            },
            {
                "name": "distilgpt2",
                "size": "82M",
                "description": "Distilled GPT-2 model",
                "architecture": "gpt2",
                "vocab_size": 50257,
                "max_length": 1024,
                "available": True,
                "source": "huggingface"
            },
            {
                "name": "custom-model",
                "size": "unknown",
                "description": "Custom trained model",
                "architecture": "custom",
                "vocab_size": 30000,
                "max_length": 512,
                "available": False,
                "source": "local"
            }
        ]
    
    @staticmethod
    def get_model_checkpoint() -> Dict[str, Any]:
        """Get mock model checkpoint data."""
        return {
            "model_state": {
                "embedding.weight": "tensor_data",
                "transformer.layers.0.attention.weight": "tensor_data",
                "lm_head.weight": "tensor_data"
            },
            "optimizer_state": {
                "state": {},
                "param_groups": []
            },
            "config": {
                "vocab_size": 50000,
                "embedding_dim": 768,
                "num_layers": 12,
                "num_heads": 12
            },
            "training_stats": {
                "step": 1000,
                "epoch": 2,
                "loss": 2.5,
                "learning_rate": 5e-5,
                "training_time": 3600.0
            },
            "metadata": {
                "created_at": "2024-01-15T10:30:00Z",
                "framework": "pytorch",
                "version": "1.0.0"
            }
        }


# Convenience functions for common test scenarios
def create_test_workspace() -> Path:
    """Create a complete test workspace."""
    manager = TestFileManager()
    workspace = manager.create_temp_directory()
    
    # Create project structure
    directories = manager.create_test_project_structure(workspace)
    
    # Add sample data
    generator = TestDataGenerator()
    
    # Create various datasets
    sample_data = generator.create_sample_jsonl_dataset(50)
    manager.write_jsonl_file(directories['data_raw'] / 'sample.jsonl', sample_data)
    
    conversation_data = generator.create_conversation_dataset(20)
    manager.write_jsonl_file(directories['data_finetune'] / 'conversations.jsonl', conversation_data)
    
    # Create config file
    config_data = MockConfigData.get_default_config()
    manager.write_json_file(workspace / 'config.json', config_data)
    
    return workspace


def create_minimal_test_project() -> Path:
    """Create minimal test project for quick tests."""
    manager = TestFileManager()
    workspace = manager.create_temp_directory()
    
    # Create basic structure
    (workspace / 'data' / 'raw').mkdir(parents=True)
    (workspace / 'data' / 'cleaned').mkdir(parents=True)
    
    # Add minimal data
    generator = TestDataGenerator()
    sample_data = generator.create_sample_jsonl_dataset(10)
    manager.write_jsonl_file(workspace / 'data' / 'raw' / 'test.jsonl', sample_data)
    
    # Add minimal config
    config_data = MockConfigData.get_minimal_config()
    manager.write_json_file(workspace / 'config.json', config_data)
    
    return workspace