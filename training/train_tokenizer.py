"""
Enhanced tokenizer training with multiple backends and preset configurations.
Supports HuggingFace tokenizers and SentencePiece with flexible configuration options.
"""

import os
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set
from dataclasses import dataclass, asdict, field
from loguru import logger
import time
import re

# HuggingFace tokenizers
try:
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
    from tokenizers.processors import BertProcessing
    TOKENIZERS_AVAILABLE = True
except ImportError:
    logger.warning("HuggingFace tokenizers not available. HF tokenizer training will be disabled.")
    TOKENIZERS_AVAILABLE = False

# SentencePiece
try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    logger.warning("SentencePiece not available. SentencePiece tokenizer training will be disabled.")
    SENTENCEPIECE_AVAILABLE = False


@dataclass
class TokenizerConfig:
    """Enhanced configuration for tokenizer training with validation and serialization."""
    tokenizer_type: str  # "huggingface", "sentencepiece"
    vocab_size: int
    model_type: str  # "bpe", "unigram", "word", "char"
    special_tokens: Dict[str, str]
    normalization: bool = True
    lowercase: bool = False
    pre_tokenizers: List[str] = field(default_factory=list)
    post_processors: List[str] = field(default_factory=list)
    trainer_args: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None  # Configuration name for identification
    description: Optional[str] = None  # Human-readable description
    
    # Validation constants
    VALID_TOKENIZER_TYPES: Set[str] = field(default_factory=lambda: {"huggingface", "sentencepiece"}, init=False)
    VALID_MODEL_TYPES: Set[str] = field(default_factory=lambda: {"bpe", "unigram", "word", "char"}, init=False)
    VALID_HF_PRE_TOKENIZERS: Set[str] = field(default_factory=lambda: {
        "whitespace", "punctuation", "digits", "byte_level", "metaspace", "char_delimiter"
    }, init=False)
    VALID_HF_POST_PROCESSORS: Set[str] = field(default_factory=lambda: {
        "bert", "roberta", "byte_level", "template"
    }, init=False)
    VALID_SPECIAL_TOKEN_KEYS: Set[str] = field(default_factory=lambda: {
        "pad_token", "unk_token", "bos_token", "eos_token", "cls_token", 
        "sep_token", "mask_token", "additional_special_tokens"
    }, init=False)
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Validate configuration
        self.validate()
        
        # Set defaults if needed
        if not self.pre_tokenizers:
            self.pre_tokenizers = []
        if not self.post_processors:
            self.post_processors = []
        if not self.trainer_args:
            self.trainer_args = {}
    
    def validate(self) -> None:
        """Comprehensive validation of tokenizer configuration."""
        errors = []
        
        # Validate tokenizer type
        if self.tokenizer_type not in self.VALID_TOKENIZER_TYPES:
            errors.append(f"Invalid tokenizer_type '{self.tokenizer_type}'. Must be one of: {self.VALID_TOKENIZER_TYPES}")
        
        # Validate model type
        if self.model_type not in self.VALID_MODEL_TYPES:
            errors.append(f"Invalid model_type '{self.model_type}'. Must be one of: {self.VALID_MODEL_TYPES}")
        
        # Validate vocab size
        if not isinstance(self.vocab_size, int) or self.vocab_size <= 0:
            errors.append(f"vocab_size must be a positive integer, got: {self.vocab_size}")
        
        if self.vocab_size < 100:
            errors.append(f"vocab_size {self.vocab_size} is too small (minimum: 100)")
        
        if self.vocab_size > 1000000:
            errors.append(f"vocab_size {self.vocab_size} is too large (maximum: 1,000,000)")
        
        # Validate special tokens
        if not isinstance(self.special_tokens, dict):
            errors.append("special_tokens must be a dictionary")
        else:
            # Check for required tokens
            if self.tokenizer_type == "huggingface" and "unk_token" not in self.special_tokens:
                errors.append("HuggingFace tokenizers require 'unk_token' in special_tokens")
            
            if self.tokenizer_type == "sentencepiece" and "unk_token" not in self.special_tokens:
                errors.append("SentencePiece tokenizers require 'unk_token' in special_tokens")
            
            # Validate special token keys
            for key in self.special_tokens.keys():
                if key not in self.VALID_SPECIAL_TOKEN_KEYS:
                    errors.append(f"Unknown special token key '{key}'. Valid keys: {self.VALID_SPECIAL_TOKEN_KEYS}")
            
            # Validate special token values
            for key, value in self.special_tokens.items():
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"Special token '{key}' must be a non-empty string, got: {value}")
                
                # Check for valid token format (allow commas for lists like additional_special_tokens)
                if not re.match(r'^[<\[\w\]>/_,-]+$', value):
                    errors.append(f"Special token '{key}' has invalid format: {value}")
        
        # Validate pre-tokenizers for HuggingFace
        if self.tokenizer_type == "huggingface":
            for pre_tok in self.pre_tokenizers:
                if pre_tok not in self.VALID_HF_PRE_TOKENIZERS:
                    errors.append(f"Invalid HuggingFace pre-tokenizer '{pre_tok}'. Valid options: {self.VALID_HF_PRE_TOKENIZERS}")
        
        # Validate post-processors for HuggingFace
        if self.tokenizer_type == "huggingface":
            for post_proc in self.post_processors:
                if post_proc not in self.VALID_HF_POST_PROCESSORS:
                    errors.append(f"Invalid HuggingFace post-processor '{post_proc}'. Valid options: {self.VALID_HF_POST_PROCESSORS}")
        
        # Validate model type compatibility
        if self.tokenizer_type == "huggingface" and self.model_type == "char":
            errors.append("HuggingFace tokenizers do not support 'char' model type")
        
        if self.tokenizer_type == "sentencepiece" and self.model_type in ["word", "char"]:
            errors.append(f"SentencePiece does not support '{self.model_type}' model type")
        
        # Validate trainer args
        if self.tokenizer_type == "sentencepiece":
            self._validate_sentencepiece_args(errors)
        elif self.tokenizer_type == "huggingface":
            self._validate_huggingface_args(errors)
        
        # Raise validation error if any issues found
        if errors:
            raise ValueError(f"TokenizerConfig validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
    
    def _validate_sentencepiece_args(self, errors: List[str]) -> None:
        """Validate SentencePiece-specific trainer arguments."""
        if "character_coverage" in self.trainer_args:
            coverage = self.trainer_args["character_coverage"]
            if not isinstance(coverage, (int, float)) or not (0.0 < coverage <= 1.0):
                errors.append(f"character_coverage must be between 0.0 and 1.0, got: {coverage}")
        
        if "max_sentence_length" in self.trainer_args:
            max_len = self.trainer_args["max_sentence_length"]
            if not isinstance(max_len, int) or max_len <= 0:
                errors.append(f"max_sentence_length must be a positive integer, got: {max_len}")
        
        if "num_threads" in self.trainer_args:
            threads = self.trainer_args["num_threads"]
            if not isinstance(threads, int) or threads <= 0:
                errors.append(f"num_threads must be a positive integer, got: {threads}")
    
    def _validate_huggingface_args(self, errors: List[str]) -> None:
        """Validate HuggingFace-specific trainer arguments."""
        if "min_frequency" in self.trainer_args:
            min_freq = self.trainer_args["min_frequency"]
            if not isinstance(min_freq, int) or min_freq < 0:
                errors.append(f"min_frequency must be a non-negative integer, got: {min_freq}")
        
        if "continuing_subword_prefix" in self.trainer_args:
            prefix = self.trainer_args["continuing_subword_prefix"]
            if not isinstance(prefix, str):
                errors.append(f"continuing_subword_prefix must be a string, got: {prefix}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        config_dict = asdict(self)
        
        # Remove validation constants from serialization
        config_dict.pop("VALID_TOKENIZER_TYPES", None)
        config_dict.pop("VALID_MODEL_TYPES", None)
        config_dict.pop("VALID_HF_PRE_TOKENIZERS", None)
        config_dict.pop("VALID_HF_POST_PROCESSORS", None)
        config_dict.pop("VALID_SPECIAL_TOKEN_KEYS", None)
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TokenizerConfig':
        """Create configuration from dictionary."""
        # Remove any validation constants that might be in the dict
        clean_dict = {k: v for k, v in config_dict.items() 
                     if not k.startswith("VALID_")}
        
        return cls(**clean_dict)
    
    def save(self, file_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved TokenizerConfig to {file_path}")
    
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'TokenizerConfig':
        """Load configuration from JSON file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = cls.from_dict(config_dict)
        logger.info(f"Loaded TokenizerConfig from {file_path}")
        return config
    
    def copy(self, **overrides) -> 'TokenizerConfig':
        """Create a copy of the configuration with optional overrides."""
        config_dict = self.to_dict()
        config_dict.update(overrides)
        return self.from_dict(config_dict)
    
    def is_compatible_with(self, other: 'TokenizerConfig') -> bool:
        """Check if this configuration is compatible with another."""
        return (
            self.tokenizer_type == other.tokenizer_type and
            self.model_type == other.model_type and
            self.vocab_size == other.vocab_size and
            self.special_tokens == other.special_tokens
        )
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the configuration."""
        summary = []
        summary.append(f"TokenizerConfig: {self.name or 'Unnamed'}")
        if self.description:
            summary.append(f"Description: {self.description}")
        summary.append(f"Type: {self.tokenizer_type} ({self.model_type})")
        summary.append(f"Vocabulary size: {self.vocab_size:,}")
        summary.append(f"Special tokens: {len(self.special_tokens)}")
        if self.pre_tokenizers:
            summary.append(f"Pre-tokenizers: {', '.join(self.pre_tokenizers)}")
        if self.post_processors:
            summary.append(f"Post-processors: {', '.join(self.post_processors)}")
        summary.append(f"Normalization: {'enabled' if self.normalization else 'disabled'}")
        summary.append(f"Lowercase: {'enabled' if self.lowercase else 'disabled'}")
        
        return "\n".join(summary)


class TokenizerTrainer(ABC):
    """Abstract base class for tokenizer training."""
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.tokenizer = None
        self.training_stats = {}
    
    @abstractmethod
    def train(self, input_files: List[Path], output_dir: Path) -> bool:
        """Train tokenizer and save artifacts."""
        pass
    
    @abstractmethod
    def load_tokenizer(self, tokenizer_path: Path) -> bool:
        """Load trained tokenizer."""
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        pass
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.config.vocab_size
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return self.training_stats
    
    @staticmethod
    def load_presets() -> Dict[str, TokenizerConfig]:
        """Load comprehensive preset configurations for common use cases."""
        presets = {
            # GPT-style configurations
            "gpt2_small": TokenizerConfig(
                name="GPT-2 Small",
                description="Small GPT-2 style tokenizer with 16K vocabulary",
                tokenizer_type="huggingface",
                vocab_size=16000,
                model_type="bpe",
                special_tokens={
                    "pad_token": "<pad>",
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>"
                },
                normalization=True,
                lowercase=False,
                pre_tokenizers=["whitespace", "punctuation"],
                trainer_args={"min_frequency": 2}
            ),
            
            "gpt2_medium": TokenizerConfig(
                name="GPT-2 Medium",
                description="Medium GPT-2 style tokenizer with 32K vocabulary",
                tokenizer_type="huggingface",
                vocab_size=32000,
                model_type="bpe",
                special_tokens={
                    "pad_token": "<pad>",
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>"
                },
                normalization=True,
                lowercase=False,
                pre_tokenizers=["whitespace", "punctuation"],
                trainer_args={"min_frequency": 2}
            ),
            
            "gpt2_large": TokenizerConfig(
                name="GPT-2 Large",
                description="Large GPT-2 style tokenizer with 50K vocabulary",
                tokenizer_type="huggingface",
                vocab_size=50000,
                model_type="bpe",
                special_tokens={
                    "pad_token": "<pad>",
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>"
                },
                normalization=True,
                lowercase=False,
                pre_tokenizers=["whitespace", "punctuation"],
                trainer_args={"min_frequency": 2}
            ),
            
            # BERT-style configurations
            "bert_base": TokenizerConfig(
                name="BERT Base",
                description="BERT-style tokenizer with WordPiece and special tokens",
                tokenizer_type="huggingface",
                vocab_size=30000,
                model_type="bpe",
                special_tokens={
                    "pad_token": "[PAD]",
                    "unk_token": "[UNK]",
                    "cls_token": "[CLS]",
                    "sep_token": "[SEP]",
                    "mask_token": "[MASK]"
                },
                normalization=True,
                lowercase=True,
                pre_tokenizers=["whitespace", "punctuation"],
                post_processors=["bert"],
                trainer_args={"min_frequency": 2}
            ),
            
            "bert_large": TokenizerConfig(
                name="BERT Large",
                description="Large BERT-style tokenizer with extended vocabulary",
                tokenizer_type="huggingface",
                vocab_size=50000,
                model_type="bpe",
                special_tokens={
                    "pad_token": "[PAD]",
                    "unk_token": "[UNK]",
                    "cls_token": "[CLS]",
                    "sep_token": "[SEP]",
                    "mask_token": "[MASK]"
                },
                normalization=True,
                lowercase=True,
                pre_tokenizers=["whitespace", "punctuation"],
                post_processors=["bert"],
                trainer_args={"min_frequency": 2}
            ),
            
            # RoBERTa-style configurations
            "roberta_base": TokenizerConfig(
                name="RoBERTa Base",
                description="RoBERTa-style tokenizer with byte-level BPE",
                tokenizer_type="huggingface",
                vocab_size=32000,
                model_type="bpe",
                special_tokens={
                    "pad_token": "<pad>",
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>"
                },
                normalization=True,
                lowercase=False,
                pre_tokenizers=["byte_level"],
                post_processors=["byte_level"],
                trainer_args={"min_frequency": 2}
            ),
            
            # SentencePiece configurations
            "sentencepiece_unigram_small": TokenizerConfig(
                name="SentencePiece Unigram Small",
                description="Small SentencePiece Unigram tokenizer for efficient training",
                tokenizer_type="sentencepiece",
                vocab_size=16000,
                model_type="unigram",
                special_tokens={
                    "pad_token": "<pad>",
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>"
                },
                normalization=True,
                trainer_args={
                    "character_coverage": 0.9995,
                    "split_by_whitespace": True,
                    "max_sentence_length": 4192,
                    "num_threads": 1
                }
            ),
            
            "sentencepiece_unigram_large": TokenizerConfig(
                name="SentencePiece Unigram Large",
                description="Large SentencePiece Unigram tokenizer for comprehensive coverage",
                tokenizer_type="sentencepiece",
                vocab_size=64000,
                model_type="unigram",
                special_tokens={
                    "pad_token": "<pad>",
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>"
                },
                normalization=True,
                trainer_args={
                    "character_coverage": 0.9999,
                    "split_by_whitespace": True,
                    "max_sentence_length": 8192,
                    "num_threads": 1
                }
            ),
            
            "sentencepiece_bpe_small": TokenizerConfig(
                name="SentencePiece BPE Small",
                description="Small SentencePiece BPE tokenizer for fast training",
                tokenizer_type="sentencepiece",
                vocab_size=16000,
                model_type="bpe",
                special_tokens={
                    "pad_token": "<pad>",
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>"
                },
                normalization=True,
                trainer_args={
                    "character_coverage": 0.9995,
                    "split_by_whitespace": True,
                    "max_sentence_length": 4192,
                    "num_threads": 1
                }
            ),
            
            "sentencepiece_bpe_large": TokenizerConfig(
                name="SentencePiece BPE Large",
                description="Large SentencePiece BPE tokenizer for comprehensive vocabulary",
                tokenizer_type="sentencepiece",
                vocab_size=64000,
                model_type="bpe",
                special_tokens={
                    "pad_token": "<pad>",
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>"
                },
                normalization=True,
                trainer_args={
                    "character_coverage": 0.9999,
                    "split_by_whitespace": True,
                    "max_sentence_length": 8192,
                    "num_threads": 1
                }
            ),
            
            # Specialized configurations
            "code_tokenizer": TokenizerConfig(
                name="Code Tokenizer",
                description="Specialized tokenizer for source code with programming tokens",
                tokenizer_type="huggingface",
                vocab_size=32000,
                model_type="bpe",
                special_tokens={
                    "pad_token": "<pad>",
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "additional_special_tokens": "<indent>,<dedent>,<newline>"
                },
                normalization=False,  # Preserve code formatting
                lowercase=False,
                pre_tokenizers=["whitespace"],  # Don't split on punctuation for code
                trainer_args={"min_frequency": 1}  # Lower frequency for code tokens
            ),
            
            "multilingual_tokenizer": TokenizerConfig(
                name="Multilingual Tokenizer",
                description="Tokenizer optimized for multiple languages with high character coverage",
                tokenizer_type="sentencepiece",
                vocab_size=128000,
                model_type="unigram",
                special_tokens={
                    "pad_token": "<pad>",
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>"
                },
                normalization=True,
                trainer_args={
                    "character_coverage": 0.99999,  # Very high coverage for multilingual
                    "split_by_whitespace": True,
                    "split_by_unicode_script": True,
                    "max_sentence_length": 8192,
                    "num_threads": 1
                }
            ),
            
            "chat_tokenizer": TokenizerConfig(
                name="Chat Tokenizer",
                description="Tokenizer optimized for conversational AI with chat-specific tokens",
                tokenizer_type="huggingface",
                vocab_size=32000,
                model_type="bpe",
                special_tokens={
                    "pad_token": "<pad>",
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "additional_special_tokens": "<user>,<assistant>,<system>"
                },
                normalization=True,
                lowercase=False,
                pre_tokenizers=["whitespace", "punctuation"],
                trainer_args={"min_frequency": 2}
            )
        }
        
        return presets
    
    @staticmethod
    def list_presets() -> List[str]:
        """Get list of available preset names."""
        return list(TokenizerTrainer.load_presets().keys())
    
    @staticmethod
    def get_preset(name: str) -> 'TokenizerConfig':
        """Get a specific preset configuration by name."""
        presets = TokenizerTrainer.load_presets()
        if name not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(f"Unknown preset '{name}'. Available presets: {available}")
        return presets[name]
    
    @staticmethod
    def save_preset(config: 'TokenizerConfig', name: str, presets_dir: Union[str, Path] = None) -> None:
        """Save a configuration as a preset."""
        if presets_dir is None:
            presets_dir = Path.cwd() / "tokenizer_presets"
        else:
            presets_dir = Path(presets_dir)
        
        presets_dir.mkdir(parents=True, exist_ok=True)
        preset_file = presets_dir / f"{name}.json"
        
        # Set the name in the config
        config_copy = config.copy(name=name)
        config_copy.save(preset_file)
        
        logger.info(f"Saved preset '{name}' to {preset_file}")
    
    @staticmethod
    def load_preset_from_file(file_path: Union[str, Path]) -> 'TokenizerConfig':
        """Load a preset configuration from file."""
        return TokenizerConfig.load(file_path)


class HuggingFaceTokenizerTrainer(TokenizerTrainer):
    """HuggingFace tokenizers implementation."""
    
    def __init__(self, config: TokenizerConfig):
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("HuggingFace tokenizers not available")
        
        super().__init__(config)
        self.name = "HuggingFaceTokenizerTrainer"
        
    def _create_tokenizer(self) -> Tokenizer:
        """Create tokenizer based on configuration."""
        # Initialize model
        if self.config.model_type == "bpe":
            model = models.BPE(unk_token=self.config.special_tokens.get("unk_token", "<unk>"))
        elif self.config.model_type == "unigram":
            model = models.Unigram()
        elif self.config.model_type == "word":
            model = models.WordLevel(unk_token=self.config.special_tokens.get("unk_token", "<unk>"))
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        tokenizer = Tokenizer(model)
        
        # Add normalization
        if self.config.normalization:
            normalizer_sequence = []
            if self.config.lowercase:
                normalizer_sequence.append(normalizers.Lowercase())
            normalizer_sequence.append(normalizers.NFD())
            normalizer_sequence.append(normalizers.StripAccents())
            
            if len(normalizer_sequence) > 1:
                tokenizer.normalizer = normalizers.Sequence(normalizer_sequence)
            elif len(normalizer_sequence) == 1:
                tokenizer.normalizer = normalizer_sequence[0]
        
        # Add pre-tokenizers
        pre_tok_sequence = []
        for pre_tok_name in self.config.pre_tokenizers:
            if pre_tok_name == "whitespace":
                pre_tok_sequence.append(pre_tokenizers.Whitespace())
            elif pre_tok_name == "punctuation":
                pre_tok_sequence.append(pre_tokenizers.Punctuation())
            elif pre_tok_name == "digits":
                pre_tok_sequence.append(pre_tokenizers.Digits())
        
        if pre_tok_sequence:
            if len(pre_tok_sequence) > 1:
                tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pre_tok_sequence)
            else:
                tokenizer.pre_tokenizer = pre_tok_sequence[0]
        
        # Add decoder
        if self.config.model_type == "bpe":
            tokenizer.decoder = decoders.BPEDecoder()
        elif self.config.model_type == "word":
            tokenizer.decoder = decoders.WordPiece()
        
        return tokenizer
    
    def _create_trainer(self) -> trainers.Trainer:
        """Create trainer based on configuration."""
        special_tokens = list(self.config.special_tokens.values())
        
        if self.config.model_type == "bpe":
            trainer = trainers.BpeTrainer(
                vocab_size=self.config.vocab_size,
                special_tokens=special_tokens,
                min_frequency=self.config.trainer_args.get("min_frequency", 2),
                continuing_subword_prefix=self.config.trainer_args.get("continuing_subword_prefix", "##"),
                end_of_word_suffix=self.config.trainer_args.get("end_of_word_suffix", "")
            )
        elif self.config.model_type == "unigram":
            trainer = trainers.UnigramTrainer(
                vocab_size=self.config.vocab_size,
                special_tokens=special_tokens,
                unk_token=self.config.special_tokens.get("unk_token", "<unk>")
            )
        elif self.config.model_type == "word":
            trainer = trainers.WordLevelTrainer(
                vocab_size=self.config.vocab_size,
                special_tokens=special_tokens,
                min_frequency=self.config.trainer_args.get("min_frequency", 2)
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        return trainer
    
    def train(self, input_files: List[Path], output_dir: Path) -> bool:
        """Train HuggingFace tokenizer."""
        try:
            logger.info(f"Training HuggingFace tokenizer with {len(input_files)} files")
            logger.info(f"Config: {self.config.model_type}, vocab_size={self.config.vocab_size}")
            
            start_time = time.time()
            
            # Create tokenizer and trainer
            tokenizer = self._create_tokenizer()
            trainer = self._create_trainer()
            
            # Convert paths to strings
            file_paths = [str(f) for f in input_files]
            
            # Train tokenizer
            tokenizer.train(file_paths, trainer)
            
            # Add post-processors
            if "bert" in self.config.post_processors:
                cls_token = self.config.special_tokens.get("cls_token", "[CLS]")
                sep_token = self.config.special_tokens.get("sep_token", "[SEP]")
                tokenizer.post_processor = BertProcessing(
                    (sep_token, tokenizer.token_to_id(sep_token)),
                    (cls_token, tokenizer.token_to_id(cls_token))
                )
            
            # Save tokenizer
            output_dir.mkdir(parents=True, exist_ok=True)
            tokenizer_path = output_dir / "tokenizer.json"
            tokenizer.save(str(tokenizer_path))
            
            # Save configuration
            config_path = output_dir / "tokenizer_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            
            # Save vocabulary
            vocab_path = output_dir / "vocab.json"
            vocab = tokenizer.get_vocab()
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(vocab, f, indent=2, ensure_ascii=False)
            
            # Store tokenizer and stats
            self.tokenizer = tokenizer
            training_time = time.time() - start_time
            
            self.training_stats = {
                'training_time': training_time,
                'vocab_size': len(vocab),
                'input_files': len(input_files),
                'model_type': self.config.model_type,
                'special_tokens': self.config.special_tokens
            }
            
            logger.info(f"HuggingFace tokenizer training completed in {training_time:.2f}s")
            logger.info(f"Vocabulary size: {len(vocab)}")
            logger.info(f"Saved to: {output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training HuggingFace tokenizer: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_tokenizer(self, tokenizer_path: Path) -> bool:
        """Load trained HuggingFace tokenizer."""
        try:
            if tokenizer_path.is_dir():
                tokenizer_file = tokenizer_path / "tokenizer.json"
            else:
                tokenizer_file = tokenizer_path
            
            if not tokenizer_file.exists():
                logger.error(f"Tokenizer file not found: {tokenizer_file}")
                return False
            
            self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
            logger.info(f"Loaded HuggingFace tokenizer from {tokenizer_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading HuggingFace tokenizer: {e}")
            return False
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        
        encoding = self.tokenizer.encode(text)
        return encoding.ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        
        return self.tokenizer.decode(token_ids)
    
    def get_vocab_size(self) -> int:
        """Get actual vocabulary size."""
        if self.tokenizer is None:
            return self.config.vocab_size
        
        return self.tokenizer.get_vocab_size()


class SentencePieceTokenizerTrainer(TokenizerTrainer):
    """SentencePiece tokenizer implementation with enhanced CLI integration."""
    
    def __init__(self, config: TokenizerConfig):
        if not SENTENCEPIECE_AVAILABLE:
            raise ImportError("SentencePiece not available")
        
        super().__init__(config)
        self.name = "SentencePieceTokenizerTrainer"
        self.sp_model = None
        
        # Validate configuration for SentencePiece
        self._validate_config()
    
    def _validate_config(self):
        """Validate SentencePiece-specific configuration."""
        # Ensure vocab_size is reasonable for the corpus size
        if self.config.vocab_size < 100:
            logger.warning(f"Very small vocab_size ({self.config.vocab_size}) may cause training issues")
        
        # Set default trainer args if not provided
        if not self.config.trainer_args:
            self.config.trainer_args = {}
        
        # Set sensible defaults
        defaults = {
            'character_coverage': 0.9995,
            'split_by_whitespace': True,
            'max_sentence_length': 4192,
            'num_threads': 1,  # Reduce threads to avoid issues
            'input_sentence_size': 0,  # Use all sentences
            'shuffle_input_sentence': True
        }
        
        for key, value in defaults.items():
            if key not in self.config.trainer_args:
                self.config.trainer_args[key] = value
    
    def train(self, input_files: List[Path], output_dir: Path) -> bool:
        """Train SentencePiece tokenizer with enhanced CLI integration."""
        try:
            logger.info(f"Training SentencePiece tokenizer with {len(input_files)} files")
            logger.info(f"Config: {self.config.model_type}, vocab_size={self.config.vocab_size}")
            
            start_time = time.time()
            
            # Validate input files and ensure text content
            valid_files = []
            total_size = 0
            for file_path in input_files:
                if file_path.exists() and file_path.stat().st_size > 0:
                    # Check if it's a text file (not binary)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as test_file:
                            content = test_file.read(1000)  # Read first 1000 chars
                            if content.strip():  # Has actual content
                                valid_files.append(file_path)
                                total_size += file_path.stat().st_size
                            else:
                                logger.warning(f"Skipping file with no content: {file_path}")
                    except Exception as e:
                        logger.warning(f"Skipping unreadable file {file_path}: {e}")
                else:
                    logger.warning(f"Skipping empty or missing file: {file_path}")
            
            if not valid_files:
                raise ValueError("No valid input files found")
            
            logger.info(f"Using {len(valid_files)} valid files, total size: {total_size / 1024:.1f} KB")
            
            # Adjust vocab_size based on corpus size if needed
            adjusted_vocab_size = self._adjust_vocab_size_for_corpus(total_size)
            if adjusted_vocab_size != self.config.vocab_size:
                logger.info(f"Adjusting vocab_size from {self.config.vocab_size} to {adjusted_vocab_size} based on corpus size ({total_size/1024:.1f} KB)")
                self.config.vocab_size = adjusted_vocab_size
                
            # Ensure minimum vocabulary size for very small corpora
            min_vocab = max(50, min(1000, int(total_size / 100)))  # At least 1 vocab entry per 100 bytes
            if adjusted_vocab_size < min_vocab:
                adjusted_vocab_size = min_vocab
                logger.info(f"Further adjusting vocab_size to {adjusted_vocab_size} for minimum coverage")
                self.config.vocab_size = adjusted_vocab_size
            
            # Prepare training arguments
            # Create a single corpus file to avoid issues with spaces in file paths
            output_dir.mkdir(parents=True, exist_ok=True)
            model_prefix = str(output_dir / "sentencepiece")
            corpus_path = output_dir / "corpus.txt"
            
            total_chars = 0
            with open(corpus_path, 'w', encoding='utf-8') as corpus_out:
                for fp in valid_files:
                    try:
                        with open(fp, 'r', encoding='utf-8', errors='ignore') as fin:
                            content = fin.read()
                            if content.strip():  # Only write non-empty content
                                # Split long lines to avoid SentencePiece max_sentence_length issues
                                lines = content.split('\n')
                                for line in lines:
                                    line = line.strip()
                                    if line:  # Skip empty lines
                                        # Split very long lines into chunks
                                        max_line_length = 2000  # Conservative limit
                                        if len(line) > max_line_length:
                                            # Split at sentence boundaries or spaces
                                            chunks = []
                                            words = line.split()
                                            current_chunk = []
                                            current_length = 0
                                            
                                            for word in words:
                                                if current_length + len(word) + 1 > max_line_length:
                                                    if current_chunk:
                                                        chunks.append(' '.join(current_chunk))
                                                        current_chunk = [word]
                                                        current_length = len(word)
                                                    else:
                                                        # Single word too long, truncate it
                                                        chunks.append(word[:max_line_length])
                                                        current_chunk = []
                                                        current_length = 0
                                                else:
                                                    current_chunk.append(word)
                                                    current_length += len(word) + 1
                                            
                                            if current_chunk:
                                                chunks.append(' '.join(current_chunk))
                                            
                                            for chunk in chunks:
                                                corpus_out.write(chunk + '\n')
                                                total_chars += len(chunk)
                                        else:
                                            corpus_out.write(line + '\n')
                                            total_chars += len(line)
                                corpus_out.write("\n")  # File separation
                    except Exception as fe:
                        logger.warning(f"Failed to read {fp}: {fe}")
            
            # Check if corpus has content
            if total_chars == 0:
                raise ValueError("No valid text content found in input files")
                
            corpus_size = corpus_path.stat().st_size
            if corpus_size == 0:
                raise ValueError("Created corpus file is empty")
                
            logger.info(f"Created corpus with {total_chars:,} characters")
            # Use forward slashes for cross-platform compatibility
            input_files_str = str(corpus_path).replace('\\', '/')
            
            # Build comprehensive training parameters (kwargs)
            train_params = self._build_training_params(input_files_str, model_prefix)
            
            # Log the training parameters for debugging (mask long input list)
            logger.debug("SentencePiece training params: " + 
                         ", ".join([f"{k}={('<paths>' if k=='input' else v)}" for k, v in train_params.items()]))
            
            # Train model with error handling using kwargs to avoid quoting issues
            try:
                spm.SentencePieceTrainer.train(**train_params)
            except RuntimeError as e:
                error_msg = str(e)
                if "Vocabulary size too high" in error_msg:
                    # Extract suggested vocab size and retry
                    import re
                    match = re.search(r'Please set it to a value <= (\d+)', error_msg)
                    if match:
                        max_vocab = int(match.group(1))
                        logger.warning(f"Vocab size too high, retrying with {max_vocab}")
                        self.config.vocab_size = max_vocab
                        train_params = self._build_training_params(input_files_str, model_prefix)
                        spm.SentencePieceTrainer.train(**train_params)
                    else:
                        raise
                else:
                    raise
            
            # Load trained model
            self.sp_model = spm.SentencePieceProcessor()
            model_file = f"{model_prefix}.model"
            
            if not Path(model_file).exists():
                raise FileNotFoundError(f"Model file not created: {model_file}")
            
            self.sp_model.load(model_file)
            
            # Save configuration and artifacts
            self._save_artifacts(output_dir)
            
            # Store stats
            training_time = time.time() - start_time
            actual_vocab_size = self.sp_model.get_piece_size()
            
            self.training_stats = {
                'training_time': training_time,
                'vocab_size': actual_vocab_size,
                'requested_vocab_size': self.config.vocab_size,
                'input_files': len(valid_files),
                'corpus_size_kb': total_size / 1024,
                'model_type': self.config.model_type,
                'special_tokens': self.config.special_tokens,
                'pieces_per_second': actual_vocab_size / training_time if training_time > 0 else 0
            }
            
            logger.info(f"SentencePiece tokenizer training completed in {training_time:.2f}s")
            logger.info(f"Vocabulary size: {actual_vocab_size} (requested: {self.config.vocab_size})")
            logger.info(f"Training speed: {self.training_stats['pieces_per_second']:.1f} pieces/second")
            logger.info(f"Saved to: {output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training SentencePiece tokenizer: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _adjust_vocab_size_for_corpus(self, corpus_size_bytes: int) -> int:
        """Adjust vocabulary size based on corpus size to avoid training failures."""
        # Rule of thumb: vocab_size should be much smaller than unique tokens in corpus
        # For small corpora, use conservative estimates
        
        if corpus_size_bytes < 1024:  # < 1KB
            return min(self.config.vocab_size, 50)
        elif corpus_size_bytes < 10 * 1024:  # < 10KB
            return min(self.config.vocab_size, 200)
        elif corpus_size_bytes < 100 * 1024:  # < 100KB
            return min(self.config.vocab_size, 1000)
        elif corpus_size_bytes < 1024 * 1024:  # < 1MB
            return min(self.config.vocab_size, 5000)
        else:
            return self.config.vocab_size  # Use requested size for larger corpora
    
    def _build_training_params(self, input_files_str: str, model_prefix: str) -> Dict[str, Any]:
        """Build comprehensive SentencePiece training kwargs to avoid quoting issues."""
        params: Dict[str, Any] = {
            "input": input_files_str,
            "model_prefix": model_prefix,
            "vocab_size": self.config.vocab_size,
            "model_type": self.config.model_type,
            "normalization_rule_name": "nfkc_cf" if self.config.normalization else "identity",
        }
        
        # Special tokens handling
        control_symbols: List[str] = []
        unk_piece: Optional[str] = None
        for key, token in self.config.special_tokens.items():
            if key == "unk_token":
                unk_piece = token
            else:
                control_symbols.append(token)
        if control_symbols:
            params["control_symbols"] = ",".join(control_symbols)
        if unk_piece:
            params["unk_piece"] = unk_piece
        
        # Set default parameters for better handling of long text
        params.update({
            "character_coverage": 0.9995,
            "max_sentence_length": 8192,  # Increased from default 4192
            "num_threads": 1,
            "input_sentence_size": 0,
            "shuffle_input_sentence": True,
            "split_by_whitespace": True,
        })
        
        # Trainer-specific arguments (override defaults)
        for key, value in self.config.trainer_args.items():
            if key in {
                "character_coverage",
                "max_sentence_length",
                "num_threads",
                "input_sentence_size",
                "seed_sentencepiece_size",
                "shrinking_factor",
                "max_sentencepiece_length",
            }:
                params[key] = value
            elif key in {
                "split_by_whitespace",
                "shuffle_input_sentence",
                "split_by_unicode_script",
                "split_by_number",
                "split_digits",
                "byte_fallback",
                "hard_vocab_limit",
            }:
                params[key] = bool(value)
        
        return params
    
    def _save_artifacts(self, output_dir: Path):
        """Save tokenizer artifacts and metadata."""
        # Save configuration
        config_path = output_dir / "tokenizer_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Create vocabulary file
        vocab_path = output_dir / "vocab.txt"
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for i in range(self.sp_model.get_piece_size()):
                piece = self.sp_model.id_to_piece(i)
                f.write(f"{piece}\n")
        
        # Create vocabulary with scores
        vocab_scores_path = output_dir / "vocab_with_scores.txt"
        with open(vocab_scores_path, 'w', encoding='utf-8') as f:
            for i in range(self.sp_model.get_piece_size()):
                piece = self.sp_model.id_to_piece(i)
                score = self.sp_model.get_score(i)
                f.write(f"{piece}\t{score}\n")
        
        # Save model info
        info_path = output_dir / "model_info.json"
        model_info = {
            'vocab_size': self.sp_model.get_piece_size(),
            'model_type': self.config.model_type,
            'special_tokens': self.config.special_tokens,
            'bos_id': self.sp_model.bos_id(),
            'eos_id': self.sp_model.eos_id(),
            'unk_id': self.sp_model.unk_id(),
            'pad_id': self.sp_model.pad_id() if hasattr(self.sp_model, 'pad_id') else -1
        }
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Saved artifacts: config, vocab, vocab_with_scores, model_info")
    
    def load_tokenizer(self, tokenizer_path: Path) -> bool:
        """Load trained SentencePiece tokenizer."""
        try:
            if tokenizer_path.is_dir():
                model_file = tokenizer_path / "sentencepiece.model"
            else:
                model_file = tokenizer_path
            
            if not model_file.exists():
                logger.error(f"SentencePiece model file not found: {model_file}")
                return False
            
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load(str(model_file))
            logger.info(f"Loaded SentencePiece tokenizer from {model_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading SentencePiece tokenizer: {e}")
            return False
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.sp_model is None:
            raise ValueError("Tokenizer not trained or loaded")
        
        return self.sp_model.encode_as_ids(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if self.sp_model is None:
            raise ValueError("Tokenizer not trained or loaded")
        
        return self.sp_model.decode_ids(token_ids)
    
    def get_vocab_size(self) -> int:
        """Get actual vocabulary size."""
        if self.sp_model is None:
            return self.config.vocab_size
        
        return self.sp_model.get_piece_size()


def create_tokenizer_trainer(config: TokenizerConfig) -> TokenizerTrainer:
    """Factory function to create tokenizer trainer."""
    if config.tokenizer_type == "huggingface":
        return HuggingFaceTokenizerTrainer(config)
    elif config.tokenizer_type == "sentencepiece":
        return SentencePieceTokenizerTrainer(config)
    else:
        raise ValueError(f"Unsupported tokenizer type: {config.tokenizer_type}")


def main():
    """Main entry point for tokenizer training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train tokenizers with multiple backends")
    parser.add_argument("--input-dir", required=True, help="Input directory with text files")
    parser.add_argument("--output-dir", required=True, help="Output directory for tokenizer")
    parser.add_argument("--preset", help="Preset configuration name")
    parser.add_argument("--tokenizer-type", choices=["huggingface", "sentencepiece"], 
                       default="huggingface", help="Tokenizer backend")
    parser.add_argument("--model-type", choices=["bpe", "unigram", "word"], 
                       default="bpe", help="Model type")
    parser.add_argument("--vocab-size", type=int, default=16000, help="Vocabulary size")
    parser.add_argument("--list-presets", action="store_true", help="List available presets")
    
    args = parser.parse_args()
    
    # List presets
    if args.list_presets:
        presets = TokenizerTrainer.load_presets()
        print("Available presets:")
        for name, config in presets.items():
            print(f"  {name}: {config.tokenizer_type} {config.model_type} (vocab_size={config.vocab_size})")
        return
    
    # Load configuration
    if args.preset:
        presets = TokenizerTrainer.load_presets()
        if args.preset not in presets:
            logger.error(f"Unknown preset: {args.preset}")
            logger.info(f"Available presets: {list(presets.keys())}")
            return
        config = presets[args.preset]
        logger.info(f"Using preset: {args.preset}")
    else:
        # Create custom configuration
        config = TokenizerConfig(
            tokenizer_type=args.tokenizer_type,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            special_tokens={
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>"
            }
        )
    
    # Find input files
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    input_files = list(input_dir.glob("*.txt"))
    if not input_files:
        logger.error(f"No .txt files found in {input_dir}")
        return
    
    logger.info(f"Found {len(input_files)} input files")
    
    # Create trainer and train
    try:
        trainer = create_tokenizer_trainer(config)
        output_dir = Path(args.output_dir)
        
        if trainer.train(input_files, output_dir):
            logger.info("✅ Tokenizer training completed successfully!")
            
            # Test the tokenizer
            test_text = "Hello world! This is a test sentence."
            token_ids = trainer.encode(test_text)
            decoded_text = trainer.decode(token_ids)
            
            logger.info(f"Test encoding:")
            logger.info(f"  Input: {test_text}")
            logger.info(f"  Token IDs: {token_ids}")
            logger.info(f"  Decoded: {decoded_text}")
            logger.info(f"  Vocab size: {trainer.get_vocab_size()}")
            
        else:
            logger.error("❌ Tokenizer training failed!")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()