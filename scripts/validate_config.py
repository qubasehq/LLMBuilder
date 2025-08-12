#!/usr/bin/env python3
"""
Configuration validation script for LLMBuilder.
Validates configuration files and provides migration tools.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import jsonschema
from jsonschema import validate, ValidationError

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class ConfigValidator:
    """Configuration validation and migration tool."""
    
    def __init__(self):
        """Initialize configuration validator."""
        self.schema = self.get_config_schema()
        self.migration_rules = self.get_migration_rules()
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get JSON schema for configuration validation.
        
        Returns:
            JSON schema dictionary
        """
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "LLMBuilder Configuration",
            "type": "object",
            "required": ["model", "training", "tokenizer", "paths"],
            "properties": {
                "model": {
                    "type": "object",
                    "required": ["vocab_size", "embedding_dim", "num_layers", "num_heads"],
                    "properties": {
                        "vocab_size": {"type": "integer", "minimum": 1000, "maximum": 100000},
                        "embedding_dim": {"type": "integer", "minimum": 64, "maximum": 4096},
                        "num_layers": {"type": "integer", "minimum": 1, "maximum": 48},
                        "num_heads": {"type": "integer", "minimum": 1, "maximum": 32},
                        "hidden_dim": {"type": "integer", "minimum": 64},
                        "max_seq_length": {"type": "integer", "minimum": 64, "maximum": 8192},
                        "dropout": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "use_bias": {"type": "boolean"},
                        "tie_weights": {"type": "boolean"}
                    }
                },
                "training": {
                    "type": "object",
                    "required": ["batch_size", "learning_rate", "num_epochs"],
                    "properties": {
                        "batch_size": {"type": "integer", "minimum": 1, "maximum": 1024},
                        "learning_rate": {"type": "number", "minimum": 1e-6, "maximum": 1.0},
                        "weight_decay": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "num_epochs": {"type": "integer", "minimum": 1, "maximum": 1000},
                        "warmup_steps": {"type": "integer", "minimum": 0},
                        "gradient_clip_norm": {"type": "number", "minimum": 0.0},
                        "save_every": {"type": "integer", "minimum": 1},
                        "num_workers": {"type": "integer", "minimum": 0, "maximum": 32},
                        "pin_memory": {"type": "boolean"},
                        "prefetch_factor": {"type": "integer", "minimum": 1},
                        "use_mixed_precision": {"type": "boolean"},
                        "log_every": {"type": "integer", "minimum": 1},
                        "eval_every": {"type": "integer", "minimum": 1}
                    }
                },
                "tokenizer": {
                    "type": "object",
                    "required": ["vocab_size", "model_type"],
                    "properties": {
                        "vocab_size": {"type": "integer", "minimum": 1000, "maximum": 100000},
                        "model_type": {"type": "string", "enum": ["bpe", "unigram", "word", "char"]},
                        "character_coverage": {"type": "number", "minimum": 0.9, "maximum": 1.0},
                        "input_sentence_size": {"type": "integer", "minimum": 1000},
                        "shuffle_input_sentence": {"type": "boolean"},
                        "normalization_rule_name": {"type": "string"},
                        "remove_extra_whitespaces": {"type": "boolean"},
                        "add_dummy_prefix": {"type": "boolean"},
                        "eos_id": {"type": "integer", "minimum": 0},
                        "unk_id": {"type": "integer", "minimum": 0},
                        "bos_id": {"type": "integer", "minimum": 0},
                        "pad_id": {"type": "integer", "minimum": 0},
                        "training": {
                            "type": "object",
                            "properties": {
                                "trainer_type": {"type": "string", "enum": ["huggingface", "sentencepiece"]},
                                "preset": {"type": "string"},
                                "custom_config": {"type": "object"},
                                "validation": {
                                    "type": "object",
                                    "properties": {
                                        "enabled": {"type": "boolean"},
                                        "sample_texts": {"type": "array", "items": {"type": "string"}}
                                    }
                                },
                                "sentencepiece": {
                                    "type": "object",
                                    "properties": {
                                        "enabled": {"type": "boolean"},
                                        "model_prefix": {"type": "string"},
                                        "user_defined_symbols": {"type": "array", "items": {"type": "string"}},
                                        "control_symbols": {"type": "array", "items": {"type": "string"}},
                                        "byte_fallback": {"type": "boolean"}
                                    }
                                }
                            }
                        }
                    }
                },
                "preprocessing": {
                    "type": "object",
                    "properties": {
                        "max_samples": {"type": "integer", "minimum": 1},
                        "min_length": {"type": "integer", "minimum": 1},
                        "max_length": {"type": "integer", "minimum": 10},
                        "remove_duplicates": {"type": "boolean"},
                        "normalize_whitespace": {"type": "boolean"},
                        "remove_empty_lines": {"type": "boolean"},
                        "encoding": {"type": "string"},
                        "chunk_size": {"type": "integer", "minimum": 1000},
                        "supported_formats": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "ingestion": {
                    "type": "object",
                    "properties": {
                        "max_file_size_mb": {"type": "integer", "minimum": 1, "maximum": 1000},
                        "batch_size": {"type": "integer", "minimum": 1, "maximum": 100},
                        "parallel_workers": {"type": "integer", "minimum": 1, "maximum": 32},
                        "skip_existing": {"type": "boolean"},
                        "validate_output": {"type": "boolean"},
                        "progress_reporting": {"type": "boolean"},
                        "extractors": {
                            "type": "object",
                            "properties": {
                                "html": {
                                    "type": "object",
                                    "properties": {
                                        "enabled": {"type": "boolean"},
                                        "remove_scripts": {"type": "boolean"},
                                        "remove_styles": {"type": "boolean"},
                                        "remove_navigation": {"type": "boolean"},
                                        "preserve_links": {"type": "boolean"},
                                        "min_text_length": {"type": "integer", "minimum": 1}
                                    }
                                },
                                "epub": {
                                    "type": "object",
                                    "properties": {
                                        "enabled": {"type": "boolean"},
                                        "extract_metadata": {"type": "boolean"},
                                        "preserve_chapters": {"type": "boolean"},
                                        "skip_images": {"type": "boolean"},
                                        "min_chapter_length": {"type": "integer", "minimum": 1}
                                    }
                                },
                                "pdf": {
                                    "type": "object",
                                    "properties": {
                                        "enabled": {"type": "boolean"},
                                        "use_ocr_fallback": {"type": "boolean"},
                                        "ocr_confidence_threshold": {"type": "integer", "minimum": 0, "maximum": 100},
                                        "ocr_language": {"type": "string"},
                                        "extract_images": {"type": "boolean"},
                                        "preserve_layout": {"type": "boolean"}
                                    }
                                },
                                "markdown": {
                                    "type": "object",
                                    "properties": {
                                        "enabled": {"type": "boolean"},
                                        "preserve_formatting": {"type": "boolean"},
                                        "extract_code_blocks": {"type": "boolean"},
                                        "preserve_tables": {"type": "boolean"},
                                        "remove_html_tags": {"type": "boolean"}
                                    }
                                }
                            }
                        }
                    }
                },
                "deduplication": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "hash_deduplication": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "normalize_whitespace": {"type": "boolean"},
                                "normalize_case": {"type": "boolean"},
                                "normalize_punctuation": {"type": "boolean"},
                                "min_line_length": {"type": "integer", "minimum": 1}
                            }
                        },
                        "embedding_deduplication": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "model_name": {"type": "string"},
                                "similarity_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "batch_size": {"type": "integer", "minimum": 1, "maximum": 1024},
                                "max_documents": {"type": "integer", "minimum": 1},
                                "cache_embeddings": {"type": "boolean"}
                            }
                        },
                        "statistics": {
                            "type": "object",
                            "properties": {
                                "save_report": {"type": "boolean"},
                                "detailed_logging": {"type": "boolean"},
                                "progress_reporting": {"type": "boolean"}
                            }
                        }
                    }
                },
                "evaluation": {
                    "type": "object",
                    "properties": {
                        "batch_size": {"type": "integer", "minimum": 1},
                        "max_new_tokens": {"type": "integer", "minimum": 1, "maximum": 2048},
                        "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 1000},
                        "top_p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "do_sample": {"type": "boolean"},
                        "num_return_sequences": {"type": "integer", "minimum": 1, "maximum": 10},
                        "benchmark_iterations": {"type": "integer", "minimum": 1},
                        "perplexity_stride": {"type": "integer", "minimum": 1}
                    }
                },
                "logging": {
                    "type": "object",
                    "properties": {
                        "level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                        "format": {"type": "string"},
                        "rotation": {"type": "string"},
                        "retention": {"type": "string"},
                        "compression": {"type": "string"},
                        "colorize": {"type": "boolean"}
                    }
                },
                "paths": {
                    "type": "object",
                    "required": ["data_dir", "raw_data_dir", "cleaned_data_dir"],
                    "properties": {
                        "data_dir": {"type": "string"},
                        "raw_data_dir": {"type": "string"},
                        "cleaned_data_dir": {"type": "string"},
                        "deduped_data_dir": {"type": "string"},
                        "tokenized_data_dir": {"type": "string"},
                        "finetune_data_dir": {"type": "string"},
                        "tokenizer_dir": {"type": "string"},
                        "checkpoint_dir": {"type": "string"},
                        "gguf_dir": {"type": "string"},
                        "log_dir": {"type": "string"},
                        "export_dir": {"type": "string"},
                        "validation_dir": {"type": "string"},
                        "monitoring_dir": {"type": "string"}
                    }
                },
                "export": {
                    "type": "object",
                    "properties": {
                        "gguf": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "quantization_levels": {
                                    "type": "array",
                                    "items": {"type": "string", "enum": ["f32", "f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k"]}
                                },
                                "validate_output": {"type": "boolean"},
                                "compression": {"type": "boolean"},
                                "metadata": {
                                    "type": "object",
                                    "properties": {
                                        "include_training_stats": {"type": "boolean"},
                                        "include_model_info": {"type": "boolean"},
                                        "custom_fields": {"type": "object"}
                                    }
                                }
                            }
                        },
                        "pytorch": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "save_optimizer_state": {"type": "boolean"},
                                "save_scheduler_state": {"type": "boolean"},
                                "compress": {"type": "boolean"}
                            }
                        },
                        "onnx": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "opset_version": {"type": "integer", "minimum": 11, "maximum": 18},
                                "optimize": {"type": "boolean"},
                                "dynamic_axes": {"type": "boolean"}
                            }
                        },
                        "model_cards": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "template": {"type": "string"},
                                "auto_generate": {"type": "boolean"},
                                "include_metrics": {"type": "boolean"}
                            }
                        }
                    }
                },
                "device": {
                    "type": "object",
                    "properties": {
                        "use_cuda": {"type": "boolean"},
                        "cuda_device": {"type": "integer", "minimum": 0},
                        "use_mps": {"type": "boolean"},
                        "cpu_threads": {"type": "integer", "minimum": 0},
                        "enable_mkldnn": {"type": "boolean"},
                        "mixed_precision": {"type": "boolean"}
                    }
                },
                "optimization": {
                    "type": "object",
                    "properties": {
                        "optimizer": {"type": "string", "enum": ["adam", "adamw", "sgd", "rmsprop"]},
                        "beta1": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "beta2": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "eps": {"type": "number", "minimum": 1e-12, "maximum": 1e-3},
                        "scheduler": {"type": "string", "enum": ["linear", "cosine", "polynomial", "constant"]},
                        "min_lr": {"type": "number", "minimum": 0.0},
                        "compile_model": {"type": "boolean"},
                        "use_fused_adamw": {"type": "boolean"}
                    }
                }
            }
        }
    
    def get_migration_rules(self) -> Dict[str, Any]:
        """Get migration rules for configuration updates.
        
        Returns:
            Migration rules dictionary
        """
        return {
            "v1.0_to_v1.1": {
                "description": "Add ingestion and deduplication support",
                "migrations": [
                    {
                        "type": "add_section",
                        "path": "ingestion",
                        "default_value": {
                            "max_file_size_mb": 100,
                            "batch_size": 10,
                            "parallel_workers": 4,
                            "skip_existing": True,
                            "validate_output": True,
                            "progress_reporting": True,
                            "extractors": {
                                "html": {"enabled": True, "remove_scripts": True, "remove_styles": True},
                                "epub": {"enabled": True, "extract_metadata": True, "preserve_chapters": True},
                                "pdf": {"enabled": True, "use_ocr_fallback": True, "ocr_confidence_threshold": 60},
                                "markdown": {"enabled": True, "preserve_formatting": False, "extract_code_blocks": True}
                            }
                        }
                    },
                    {
                        "type": "add_section",
                        "path": "deduplication",
                        "default_value": {
                            "enabled": True,
                            "hash_deduplication": {"enabled": True, "normalize_whitespace": True},
                            "embedding_deduplication": {"enabled": True, "similarity_threshold": 0.85},
                            "statistics": {"save_report": True, "detailed_logging": True}
                        }
                    },
                    {
                        "type": "add_field",
                        "path": "paths.deduped_data_dir",
                        "default_value": "data/deduped"
                    },
                    {
                        "type": "add_field",
                        "path": "paths.gguf_dir",
                        "default_value": "exports/gguf"
                    },
                    {
                        "type": "update_field",
                        "path": "preprocessing.supported_formats",
                        "new_value": ["txt", "pdf", "docx", "pptx", "md", "html", "epub"]
                    }
                ]
            },
            "v1.1_to_v1.2": {
                "description": "Add enhanced export and tokenizer training support",
                "migrations": [
                    {
                        "type": "add_section",
                        "path": "export",
                        "default_value": {
                            "gguf": {"enabled": True, "quantization_levels": ["f16", "q8_0", "q4_0"]},
                            "pytorch": {"enabled": True, "save_optimizer_state": False},
                            "model_cards": {"enabled": True, "auto_generate": True}
                        }
                    },
                    {
                        "type": "add_section",
                        "path": "tokenizer.training",
                        "default_value": {
                            "trainer_type": "huggingface",
                            "preset": "default",
                            "validation": {"enabled": True}
                        }
                    }
                ]
            }
        }
    
    def validate_config(self, config_path: str) -> Dict[str, Any]:
        """Validate configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Validation results
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate against schema
            validate(instance=config, schema=self.schema)
            
            # Additional validation checks
            validation_results = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'suggestions': []
            }
            
            # Check for consistency issues
            self._check_consistency(config, validation_results)
            
            # Check for performance recommendations
            self._check_performance(config, validation_results)
            
            return validation_results
            
        except FileNotFoundError:
            return {
                'valid': False,
                'errors': [f"Configuration file not found: {config_path}"],
                'warnings': [],
                'suggestions': []
            }
        except json.JSONDecodeError as e:
            return {
                'valid': False,
                'errors': [f"Invalid JSON format: {e}"],
                'warnings': [],
                'suggestions': []
            }
        except ValidationError as e:
            return {
                'valid': False,
                'errors': [f"Schema validation error: {e.message}"],
                'warnings': [],
                'suggestions': []
            }
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Validation error: {e}"],
                'warnings': [],
                'suggestions': []
            }
    
    def _check_consistency(self, config: Dict[str, Any], results: Dict[str, Any]):
        """Check configuration consistency.
        
        Args:
            config: Configuration dictionary
            results: Validation results to update
        """
        # Check vocab size consistency
        model_vocab = config.get('model', {}).get('vocab_size', 0)
        tokenizer_vocab = config.get('tokenizer', {}).get('vocab_size', 0)
        
        if model_vocab != tokenizer_vocab:
            results['warnings'].append(
                f"Model vocab_size ({model_vocab}) doesn't match tokenizer vocab_size ({tokenizer_vocab})"
            )
        
        # Check batch size consistency
        train_batch = config.get('training', {}).get('batch_size', 0)
        eval_batch = config.get('evaluation', {}).get('batch_size', 0)
        
        if eval_batch > train_batch:
            results['warnings'].append(
                f"Evaluation batch_size ({eval_batch}) is larger than training batch_size ({train_batch})"
            )
        
        # Check path consistency
        paths = config.get('paths', {})
        data_dir = paths.get('data_dir', 'data')
        
        for path_key, path_value in paths.items():
            if path_key != 'data_dir' and path_key.endswith('_data_dir'):
                if not path_value.startswith(data_dir):
                    results['warnings'].append(
                        f"Path {path_key} ({path_value}) should be under data_dir ({data_dir})"
                    )
    
    def _check_performance(self, config: Dict[str, Any], results: Dict[str, Any]):
        """Check performance-related configuration.
        
        Args:
            config: Configuration dictionary
            results: Validation results to update
        """
        training = config.get('training', {})
        
        # Check for performance recommendations
        if training.get('num_workers', 0) == 0:
            results['suggestions'].append(
                "Consider setting num_workers > 0 for better data loading performance"
            )
        
        if not training.get('pin_memory', False):
            results['suggestions'].append(
                "Consider enabling pin_memory for GPU training performance"
            )
        
        if not training.get('use_mixed_precision', False):
            results['suggestions'].append(
                "Consider enabling mixed precision training for better GPU utilization"
            )
        
        # Check deduplication settings
        dedup = config.get('deduplication', {})
        if dedup.get('enabled', False):
            embedding_dedup = dedup.get('embedding_deduplication', {})
            if embedding_dedup.get('enabled', False):
                batch_size = embedding_dedup.get('batch_size', 32)
                if batch_size > 64:
                    results['suggestions'].append(
                        f"Embedding deduplication batch_size ({batch_size}) might be too large for memory"
                    )
    
    def migrate_config(self, config_path: str, target_version: str = "latest", 
                      backup: bool = True) -> Dict[str, Any]:
        """Migrate configuration to target version.
        
        Args:
            config_path: Path to configuration file
            target_version: Target version to migrate to
            backup: Whether to create backup of original config
            
        Returns:
            Migration results
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Create backup if requested
            if backup:
                backup_path = f"{config_path}.backup"
                with open(backup_path, 'w') as f:
                    json.dump(config, f, indent=2)
            
            # Determine current version and migration path
            current_version = config.get('version', 'v1.0')
            migration_path = self._get_migration_path(current_version, target_version)
            
            migration_results = {
                'success': True,
                'migrations_applied': [],
                'errors': [],
                'backup_created': backup_path if backup else None
            }
            
            # Apply migrations
            for version_migration in migration_path:
                try:
                    self._apply_migration(config, version_migration)
                    migration_results['migrations_applied'].append(version_migration)
                except Exception as e:
                    migration_results['errors'].append(f"Migration {version_migration} failed: {e}")
                    migration_results['success'] = False
                    break
            
            # Save migrated config
            if migration_results['success']:
                config['version'] = target_version
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
            
            return migration_results
            
        except Exception as e:
            return {
                'success': False,
                'migrations_applied': [],
                'errors': [f"Migration failed: {e}"],
                'backup_created': None
            }
    
    def _get_migration_path(self, current_version: str, target_version: str) -> List[str]:
        """Get migration path from current to target version.
        
        Args:
            current_version: Current configuration version
            target_version: Target configuration version
            
        Returns:
            List of migration steps
        """
        # Simple version mapping for now
        version_order = ['v1.0', 'v1.1', 'v1.2']
        
        if target_version == "latest":
            target_version = version_order[-1]
        
        try:
            current_idx = version_order.index(current_version)
            target_idx = version_order.index(target_version)
            
            if current_idx >= target_idx:
                return []  # No migration needed
            
            migrations = []
            for i in range(current_idx, target_idx):
                migration_key = f"{version_order[i]}_to_{version_order[i+1]}"
                if migration_key in self.migration_rules:
                    migrations.append(migration_key)
            
            return migrations
            
        except ValueError:
            return []  # Unknown version
    
    def _apply_migration(self, config: Dict[str, Any], migration_key: str):
        """Apply specific migration to configuration.
        
        Args:
            config: Configuration dictionary to modify
            migration_key: Migration rule key
        """
        migration_rule = self.migration_rules.get(migration_key, {})
        migrations = migration_rule.get('migrations', [])
        
        for migration in migrations:
            migration_type = migration['type']
            path = migration['path']
            
            if migration_type == 'add_section':
                self._set_nested_value(config, path, migration['default_value'])
            elif migration_type == 'add_field':
                if not self._has_nested_value(config, path):
                    self._set_nested_value(config, path, migration['default_value'])
            elif migration_type == 'update_field':
                self._set_nested_value(config, path, migration['new_value'])
    
    def _has_nested_value(self, config: Dict[str, Any], path: str) -> bool:
        """Check if nested value exists in configuration.
        
        Args:
            config: Configuration dictionary
            path: Dot-separated path
            
        Returns:
            True if value exists
        """
        try:
            current = config
            for key in path.split('.'):
                current = current[key]
            return True
        except (KeyError, TypeError):
            return False
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """Set nested value in configuration.
        
        Args:
            config: Configuration dictionary
            path: Dot-separated path
            value: Value to set
        """
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def generate_default_config(self, output_path: str) -> bool:
        """Generate default configuration file.
        
        Args:
            output_path: Path to save default configuration
            
        Returns:
            True if successful
        """
        try:
            default_config = {
                "version": "v1.2",
                "model": {
                    "vocab_size": 16000,
                    "embedding_dim": 384,
                    "num_layers": 6,
                    "num_heads": 6,
                    "hidden_dim": 1536,
                    "max_seq_length": 256,
                    "dropout": 0.1,
                    "use_bias": True,
                    "tie_weights": True
                },
                "training": {
                    "batch_size": 4,
                    "learning_rate": 0.0002,
                    "weight_decay": 0.01,
                    "num_epochs": 5,
                    "warmup_steps": 1000,
                    "gradient_clip_norm": 1.0,
                    "save_every": 500,
                    "num_workers": 4,
                    "pin_memory": True,
                    "prefetch_factor": 2,
                    "use_mixed_precision": True,
                    "log_every": 10,
                    "eval_every": 500
                },
                "tokenizer": {
                    "vocab_size": 16000,
                    "model_type": "bpe",
                    "character_coverage": 0.9995,
                    "input_sentence_size": 10000000,
                    "shuffle_input_sentence": True,
                    "normalization_rule_name": "nmt_nfkc_cf",
                    "remove_extra_whitespaces": True,
                    "add_dummy_prefix": True,
                    "eos_id": 1,
                    "unk_id": 0,
                    "bos_id": 2,
                    "pad_id": 3,
                    "training": {
                        "trainer_type": "huggingface",
                        "preset": "default",
                        "custom_config": {},
                        "validation": {
                            "enabled": True,
                            "sample_texts": [
                                "This is a test sentence for tokenizer validation.",
                                "Another example with numbers: 123 and symbols: @#$%"
                            ]
                        },
                        "sentencepiece": {
                            "enabled": False,
                            "model_prefix": "tokenizer",
                            "user_defined_symbols": [],
                            "control_symbols": [],
                            "byte_fallback": True
                        }
                    }
                },
                "preprocessing": {
                    "max_samples": 1000,
                    "min_length": 10,
                    "max_length": 10000,
                    "remove_duplicates": True,
                    "normalize_whitespace": True,
                    "remove_empty_lines": True,
                    "encoding": "utf-8",
                    "chunk_size": 1000000,
                    "supported_formats": ["txt", "pdf", "docx", "pptx", "md", "html", "epub"]
                },
                "ingestion": {
                    "max_file_size_mb": 100,
                    "batch_size": 10,
                    "parallel_workers": 4,
                    "skip_existing": True,
                    "validate_output": True,
                    "progress_reporting": True,
                    "extractors": {
                        "html": {
                            "enabled": True,
                            "remove_scripts": True,
                            "remove_styles": True,
                            "remove_navigation": True,
                            "preserve_links": False,
                            "min_text_length": 50
                        },
                        "epub": {
                            "enabled": True,
                            "extract_metadata": True,
                            "preserve_chapters": True,
                            "skip_images": True,
                            "min_chapter_length": 100
                        },
                        "pdf": {
                            "enabled": True,
                            "use_ocr_fallback": True,
                            "ocr_confidence_threshold": 60,
                            "ocr_language": "eng",
                            "extract_images": False,
                            "preserve_layout": False
                        },
                        "markdown": {
                            "enabled": True,
                            "preserve_formatting": False,
                            "extract_code_blocks": True,
                            "preserve_tables": True,
                            "remove_html_tags": True
                        }
                    }
                },
                "deduplication": {
                    "enabled": True,
                    "hash_deduplication": {
                        "enabled": True,
                        "normalize_whitespace": True,
                        "normalize_case": True,
                        "normalize_punctuation": True,
                        "min_line_length": 10
                    },
                    "embedding_deduplication": {
                        "enabled": True,
                        "model_name": "all-MiniLM-L6-v2",
                        "similarity_threshold": 0.85,
                        "batch_size": 32,
                        "max_documents": 10000,
                        "cache_embeddings": True
                    },
                    "statistics": {
                        "save_report": True,
                        "detailed_logging": True,
                        "progress_reporting": True
                    }
                },
                "evaluation": {
                    "batch_size": 4,
                    "max_new_tokens": 100,
                    "temperature": 0.8,
                    "top_k": 50,
                    "top_p": 0.9,
                    "do_sample": True,
                    "num_return_sequences": 3,
                    "benchmark_iterations": 100,
                    "perplexity_stride": 256
                },
                "logging": {
                    "level": "INFO",
                    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
                    "rotation": "10 MB",
                    "retention": "7 days",
                    "compression": "gz",
                    "colorize": True
                },
                "paths": {
                    "data_dir": "data",
                    "raw_data_dir": "data/raw",
                    "cleaned_data_dir": "data/cleaned",
                    "deduped_data_dir": "data/deduped",
                    "tokenized_data_dir": "data/tokens",
                    "finetune_data_dir": "data/finetune",
                    "tokenizer_dir": "exports/tokenizer",
                    "checkpoint_dir": "exports/checkpoints",
                    "gguf_dir": "exports/gguf",
                    "log_dir": "logs",
                    "export_dir": "exports",
                    "validation_dir": "validation_results",
                    "monitoring_dir": "monitoring"
                },
                "export": {
                    "gguf": {
                        "enabled": True,
                        "quantization_levels": ["f16", "q8_0", "q4_0"],
                        "validate_output": True,
                        "compression": True,
                        "metadata": {
                            "include_training_stats": True,
                            "include_model_info": True,
                            "custom_fields": {}
                        }
                    },
                    "pytorch": {
                        "enabled": True,
                        "save_optimizer_state": False,
                        "save_scheduler_state": False,
                        "compress": True
                    },
                    "onnx": {
                        "enabled": False,
                        "opset_version": 14,
                        "optimize": True,
                        "dynamic_axes": True
                    },
                    "model_cards": {
                        "enabled": True,
                        "template": "templates/model_card_template.md",
                        "auto_generate": True,
                        "include_metrics": True
                    }
                },
                "device": {
                    "use_cuda": False,
                    "cuda_device": 0,
                    "use_mps": False,
                    "cpu_threads": 0,
                    "enable_mkldnn": True,
                    "mixed_precision": False
                },
                "optimization": {
                    "optimizer": "adamw",
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "eps": 1e-8,
                    "scheduler": "cosine",
                    "min_lr": 1e-6,
                    "compile_model": False,
                    "use_fused_adamw": False
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error generating default config: {e}")
            return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLMBuilder configuration validation and migration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate configuration
  python scripts/validate_config.py --validate config.json
  
  # Migrate configuration to latest version
  python scripts/validate_config.py --migrate config.json
  
  # Generate default configuration
  python scripts/validate_config.py --generate-default config_new.json
  
  # Validate and show detailed report
  python scripts/validate_config.py --validate config.json --verbose
        """
    )
    
    parser.add_argument('--validate', type=str, help='Validate configuration file')
    parser.add_argument('--migrate', type=str, help='Migrate configuration file')
    parser.add_argument('--target-version', type=str, default='latest', help='Target version for migration')
    parser.add_argument('--generate-default', type=str, help='Generate default configuration file')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup during migration')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    
    validator = ConfigValidator()
    
    try:
        if args.validate:
            print(f"Validating configuration: {args.validate}")
            results = validator.validate_config(args.validate)
            
            if results['valid']:
                print("✅ Configuration is valid!")
            else:
                print("❌ Configuration validation failed!")
                for error in results['errors']:
                    print(f"  Error: {error}")
            
            if results['warnings'] and args.verbose:
                print("\n⚠️  Warnings:")
                for warning in results['warnings']:
                    print(f"  {warning}")
            
            if results['suggestions'] and args.verbose:
                print("\n💡 Suggestions:")
                for suggestion in results['suggestions']:
                    print(f"  {suggestion}")
        
        elif args.migrate:
            print(f"Migrating configuration: {args.migrate}")
            results = validator.migrate_config(
                args.migrate, 
                target_version=args.target_version,
                backup=not args.no_backup
            )
            
            if results['success']:
                print("✅ Configuration migrated successfully!")
                if results['backup_created']:
                    print(f"📁 Backup created: {results['backup_created']}")
                for migration in results['migrations_applied']:
                    print(f"  Applied: {migration}")
            else:
                print("❌ Configuration migration failed!")
                for error in results['errors']:
                    print(f"  Error: {error}")
        
        elif args.generate_default:
            print(f"Generating default configuration: {args.generate_default}")
            if validator.generate_default_config(args.generate_default):
                print("✅ Default configuration generated successfully!")
            else:
                print("❌ Failed to generate default configuration!")
        
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n⚠️ Operation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Operation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()