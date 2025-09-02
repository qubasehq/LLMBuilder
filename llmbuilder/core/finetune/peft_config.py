"""
PEFT Configuration Management Module

This module provides configuration management and validation for Parameter-Efficient Fine-Tuning (PEFT)
using LoRA and QLoRA techniques.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
import json
import logging
from pathlib import Path

try:
    from peft import LoraConfig, TaskType
# Lazy import: from transformers import \1
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = None
    TaskType = None
    BitsAndBytesConfig = None

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """LoRA configuration parameters."""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: List[str] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["qkv_proj", "out_proj", "fc1", "fc2"]


@dataclass
class QLoRAConfig:
    """QLoRA quantization configuration parameters."""
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "float16"


@dataclass
class AdapterConfig:
    """Adapter management configuration."""
    save_directory: str = "exports/adapters"
    merge_and_unload: bool = False
    auto_save_interval: int = 500


@dataclass
class AdapterMetadata:
    """Metadata for saved adapters."""
    name: str
    base_model: str
    creation_date: str
    training_steps: int
    lora_config: LoRAConfig
    performance_metrics: Dict[str, Any]


# Mapping from GPTModel components to PEFT target modules
TARGET_MODULE_MAPPING = {
    "attention": ["qkv_proj", "out_proj"],
    "mlp": ["fc1", "fc2"],
    "all_linear": ["qkv_proj", "out_proj", "fc1", "fc2"],
    "custom": []  # User-defined
}


class PEFTConfigManager:
    """Manages PEFT-specific configuration and validation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PEFT configuration manager.
        
        Args:
            config: Full configuration dictionary containing PEFT settings
        """
        self.config = config
        self.peft_config = config.get("peft", {})
        self.validate_config()
        
        # Initialize configuration objects
        self.lora_config = self._create_lora_config()
        self.qlora_config = self._create_qlora_config()
        self.adapter_config = self._create_adapter_config()
    
    def validate_config(self) -> None:
        """Validate PEFT configuration parameters."""
        if not PEFT_AVAILABLE:
            if self.peft_config.get("enabled", False):
                raise ImportError(
                    "PEFT libraries not available. Please install: "
                    "pip install peft bitsandbytes accelerate"
                )
            return
        
        if not self.peft_config.get("enabled", False):
            return
        
        # Validate method
        method = self.peft_config.get("method", "lora")
        if method not in ["lora", "qlora"]:
            raise ValueError(f"Invalid PEFT method: {method}. Must be 'lora' or 'qlora'")
        
        # Validate LoRA parameters
        lora_config = self.peft_config.get("lora", {})
        self._validate_lora_config(lora_config)
        
        # Validate QLoRA parameters if using QLoRA
        if method == "qlora":
            qlora_config = self.peft_config.get("qlora", {})
            self._validate_qlora_config(qlora_config)
        
        # Validate adapter configuration
        adapter_config = self.peft_config.get("adapter", {})
        self._validate_adapter_config(adapter_config)
    
    def _validate_lora_config(self, lora_config: Dict[str, Any]) -> None:
        """Validate LoRA configuration parameters."""
        r = lora_config.get("r", 16)
        if not isinstance(r, int) or r < 1 or r > 256:
            raise ValueError(f"LoRA rank (r) must be an integer between 1 and 256, got: {r}")
        
        alpha = lora_config.get("alpha", 32)
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise ValueError(f"LoRA alpha must be a positive number, got: {alpha}")
        
        dropout = lora_config.get("dropout", 0.1)
        if not isinstance(dropout, (int, float)) or dropout < 0 or dropout > 1:
            raise ValueError(f"LoRA dropout must be between 0 and 1, got: {dropout}")
        
        bias = lora_config.get("bias", "none")
        if bias not in ["none", "all", "lora_only"]:
            raise ValueError(f"LoRA bias must be 'none', 'all', or 'lora_only', got: {bias}")
        
        task_type = lora_config.get("task_type", "CAUSAL_LM")
        valid_task_types = ["CAUSAL_LM", "SEQ_2_SEQ_LM", "TOKEN_CLS", "SEQ_CLS"]
        if task_type not in valid_task_types:
            raise ValueError(f"Task type must be one of {valid_task_types}, got: {task_type}")
        
        # Validate target modules
        target_modules = lora_config.get("target_modules", [])
        if not isinstance(target_modules, list):
            raise ValueError("target_modules must be a list of strings")
        
        if not target_modules:
            logger.warning("No target modules specified, using default: ['qkv_proj', 'out_proj', 'fc1', 'fc2']")
    
    def _validate_qlora_config(self, qlora_config: Dict[str, Any]) -> None:
        """Validate QLoRA configuration parameters."""
        quant_type = qlora_config.get("bnb_4bit_quant_type", "nf4")
        if quant_type not in ["fp4", "nf4"]:
            raise ValueError(f"Quantization type must be 'fp4' or 'nf4', got: {quant_type}")
        
        compute_dtype = qlora_config.get("bnb_4bit_compute_dtype", "float16")
        valid_dtypes = ["float16", "bfloat16", "float32"]
        if compute_dtype not in valid_dtypes:
            raise ValueError(f"Compute dtype must be one of {valid_dtypes}, got: {compute_dtype}")
    
    def _validate_adapter_config(self, adapter_config: Dict[str, Any]) -> None:
        """Validate adapter configuration parameters."""
        save_dir = adapter_config.get("save_directory", "exports/adapters")
        if not isinstance(save_dir, str):
            raise ValueError("save_directory must be a string")
        
        auto_save_interval = adapter_config.get("auto_save_interval", 500)
        if not isinstance(auto_save_interval, int) or auto_save_interval <= 0:
            raise ValueError("auto_save_interval must be a positive integer")
    
    def _create_lora_config(self) -> LoRAConfig:
        """Create LoRA configuration object."""
        lora_config = self.peft_config.get("lora", {})
        return LoRAConfig(
            r=lora_config.get("r", 16),
            alpha=lora_config.get("alpha", 32),
            dropout=lora_config.get("dropout", 0.1),
            target_modules=lora_config.get("target_modules", ["qkv_proj", "out_proj", "fc1", "fc2"]),
            bias=lora_config.get("bias", "none"),
            task_type=lora_config.get("task_type", "CAUSAL_LM")
        )
    
    def _create_qlora_config(self) -> QLoRAConfig:
        """Create QLoRA configuration object."""
        qlora_config = self.peft_config.get("qlora", {})
        return QLoRAConfig(
            load_in_4bit=qlora_config.get("load_in_4bit", True),
            bnb_4bit_quant_type=qlora_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=qlora_config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=qlora_config.get("bnb_4bit_compute_dtype", "float16")
        )
    
    def _create_adapter_config(self) -> AdapterConfig:
        """Create adapter configuration object."""
        adapter_config = self.peft_config.get("adapter", {})
        return AdapterConfig(
            save_directory=adapter_config.get("save_directory", "exports/adapters"),
            merge_and_unload=adapter_config.get("merge_and_unload", False),
            auto_save_interval=adapter_config.get("auto_save_interval", 500)
        )
    
    def get_lora_config(self) -> Optional[LoraConfig]:
        """Create LoRA configuration object for PEFT library.
        
        Returns:
            LoraConfig object if PEFT is available and enabled, None otherwise
        """
        if not PEFT_AVAILABLE or not self.is_enabled():
            return None
        
        # Map task type string to TaskType enum
        task_type_mapping = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
            "TOKEN_CLS": TaskType.TOKEN_CLS,
            "SEQ_CLS": TaskType.SEQ_CLS
        }
        
        return LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            target_modules=self.lora_config.target_modules,
            bias=self.lora_config.bias,
            task_type=task_type_mapping.get(self.lora_config.task_type, TaskType.CAUSAL_LM)
        )
    
    def get_bnb_config(self) -> Optional[BitsAndBytesConfig]:
        """Create quantization configuration object for bitsandbytes.
        
        Returns:
            BitsAndBytesConfig object if using QLoRA, None otherwise
        """
        if not PEFT_AVAILABLE or not self.is_qlora_enabled():
            return None
        
from llmbuilder.utils.lazy_imports import torch
        
        # Map compute dtype string to torch dtype
        dtype_mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        
        return BitsAndBytesConfig(
            load_in_4bit=self.qlora_config.load_in_4bit,
            bnb_4bit_quant_type=self.qlora_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.qlora_config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=dtype_mapping.get(
                self.qlora_config.bnb_4bit_compute_dtype, 
                torch.float16
            )
        )
    
    def is_enabled(self) -> bool:
        """Check if PEFT is enabled."""
        return self.peft_config.get("enabled", False)
    
    def is_qlora_enabled(self) -> bool:
        """Check if QLoRA is enabled."""
        return self.is_enabled() and self.peft_config.get("method") == "qlora"
    
    def get_method(self) -> str:
        """Get PEFT method (lora or qlora)."""
        return self.peft_config.get("method", "lora")
    
    def get_target_modules(self, model_architecture: str = "gpt") -> List[str]:
        """Get target modules for LoRA adaptation.
        
        Args:
            model_architecture: Model architecture type for module mapping
            
        Returns:
            List of target module names
        """
        target_modules = self.lora_config.target_modules
        
        # If using predefined mapping, expand it
        if len(target_modules) == 1 and target_modules[0] in TARGET_MODULE_MAPPING:
            mapping_key = target_modules[0]
            target_modules = TARGET_MODULE_MAPPING[mapping_key]
            logger.info(f"Expanded target modules '{mapping_key}' to: {target_modules}")
        
        return target_modules
    
    def save_adapter_metadata(self, adapter_name: str, metadata: AdapterMetadata) -> None:
        """Save adapter metadata to file.
        
        Args:
            adapter_name: Name of the adapter
            metadata: Adapter metadata object
        """
        adapter_dir = Path(self.adapter_config.save_directory) / adapter_name
        adapter_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_path = adapter_dir / "adapter_metadata.json"
        
        # Convert metadata to dictionary
        metadata_dict = {
            "name": metadata.name,
            "base_model": metadata.base_model,
            "creation_date": metadata.creation_date,
            "training_steps": metadata.training_steps,
            "lora_config": {
                "r": metadata.lora_config.r,
                "alpha": metadata.lora_config.alpha,
                "dropout": metadata.lora_config.dropout,
                "target_modules": metadata.lora_config.target_modules,
                "bias": metadata.lora_config.bias,
                "task_type": metadata.lora_config.task_type
            },
            "performance_metrics": metadata.performance_metrics
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        logger.info(f"Saved adapter metadata to: {metadata_path}")
    
    def load_adapter_metadata(self, adapter_name: str) -> Optional[AdapterMetadata]:
        """Load adapter metadata from file.
        
        Args:
            adapter_name: Name of the adapter
            
        Returns:
            AdapterMetadata object if found, None otherwise
        """
        adapter_dir = Path(self.adapter_config.save_directory) / adapter_name
        metadata_path = adapter_dir / "adapter_metadata.json"
        
        if not metadata_path.exists():
            logger.warning(f"Adapter metadata not found: {metadata_path}")
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            # Convert dictionary back to metadata object
            lora_config = LoRAConfig(**metadata_dict["lora_config"])
            
            return AdapterMetadata(
                name=metadata_dict["name"],
                base_model=metadata_dict["base_model"],
                creation_date=metadata_dict["creation_date"],
                training_steps=metadata_dict["training_steps"],
                lora_config=lora_config,
                performance_metrics=metadata_dict["performance_metrics"]
            )
        
        except Exception as e:
            logger.error(f"Failed to load adapter metadata: {e}")
            return None


def validate_peft_dependencies() -> bool:
    """Validate that PEFT dependencies are available.
    
    Returns:
        True if all dependencies are available, False otherwise
    """
    try:
        import peft
        import bitsandbytes
        import accelerate
        return True
    except ImportError as e:
        logger.error(f"PEFT dependencies not available: {e}")
        return False


def get_peft_installation_instructions() -> str:
    """Get installation instructions for PEFT dependencies.
    
    Returns:
        Installation instructions string
    """
    return """
To use PEFT functionality, please install the required dependencies:

pip install peft>=0.6.0 bitsandbytes>=0.41.0 accelerate>=0.24.0

For QLoRA support, ensure you have a CUDA-capable GPU and compatible PyTorch installation.
For CPU-only usage, LoRA is supported but QLoRA performance will be limited.
"""