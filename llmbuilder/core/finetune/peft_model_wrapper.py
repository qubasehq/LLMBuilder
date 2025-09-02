"""
PEFT Model Wrapper

This module provides a wrapper for integrating existing GPTModel with PEFT capabilities,
including LoRA and QLoRA support.
"""

import logging
from typing import Optional, Dict, Any, Union
from llmbuilder.utils.lazy_imports import torch
import torch.nn as nn
from pathlib import Path

try:
    from peft import get_peft_model, PeftModel, PeftConfig, LoraConfig
# Lazy import: from transformers import \1
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    get_peft_model = None
    PeftModel = None
    PeftConfig = None
    LoraConfig = None
    BitsAndBytesConfig = None

from .peft_config import PEFTConfigManager
from .peft_model_utils import PEFTTargetModuleValidator, validate_model_for_peft

logger = logging.getLogger(__name__)


class PEFTModelWrapper:
    """Wraps existing GPTModel with PEFT capabilities."""
    
    def __init__(self, base_model: nn.Module, peft_config_manager: PEFTConfigManager):
        """Initialize PEFT model wrapper.
        
        Args:
            base_model: Base PyTorch model (e.g., GPTModel)
            peft_config_manager: PEFT configuration manager
        """
        self.base_model = base_model
        self.peft_config_manager = peft_config_manager
        self.peft_model = None
        self.is_peft_enabled = False
        self.quantization_config = None
        
        # Validate PEFT availability
        if not PEFT_AVAILABLE and peft_config_manager.is_enabled():
            raise ImportError(
                "PEFT libraries not available. Please install: "
                "pip install peft bitsandbytes accelerate"
            )
        
        # Initialize PEFT if enabled
        if peft_config_manager.is_enabled():
            self.setup_peft_model()
    
    def setup_peft_model(self) -> None:
        """Initialize PEFT model with adapters."""
        if not PEFT_AVAILABLE:
            logger.warning("PEFT not available, skipping PEFT setup")
            return
        
        logger.info("Setting up PEFT model...")
        
        # Validate model compatibility
        if not validate_model_for_peft(self.base_model, self.peft_config_manager):
            raise ValueError("Model is not compatible with PEFT configuration")
        
        # Setup quantization if using QLoRA
        if self.peft_config_manager.is_qlora_enabled():
            self._setup_quantization()
        
        # Validate and fix target modules
        validator = PEFTTargetModuleValidator(self.base_model, self.peft_config_manager)
        target_modules = validator.validate_and_fix_target_modules()
        
        # Update target modules in config if they were fixed
        if target_modules != self.peft_config_manager.get_target_modules():
            logger.info(f"Updated target modules to: {target_modules}")
            self.peft_config_manager.lora_config.target_modules = target_modules
        
        # Create LoRA configuration
        lora_config = self.peft_config_manager.get_lora_config()
        if lora_config is None:
            raise ValueError("Failed to create LoRA configuration")
        
        # Update target modules in LoRA config
        lora_config.target_modules = target_modules
        
        # Apply PEFT to model
        try:
            self.peft_model = get_peft_model(self.base_model, lora_config)
            self.is_peft_enabled = True
            
            # Log parameter information
            self._log_parameter_info()
            
            logger.info("PEFT model setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup PEFT model: {e}")
            raise
    
    def _setup_quantization(self) -> None:
        """Setup quantization for QLoRA."""
        if not self.peft_config_manager.is_qlora_enabled():
            return
        
        logger.info("Setting up quantization for QLoRA...")
        
        # Get quantization config
        bnb_config = self.peft_config_manager.get_bnb_config()
        if bnb_config is None:
            raise ValueError("Failed to create quantization configuration")
        
        self.quantization_config = bnb_config
        
        # Apply quantization to base model
        try:
            # Note: In practice, quantization is typically applied during model loading
            # This is a placeholder for quantization setup
            logger.info("Quantization configuration prepared")
            
        except Exception as e:
            logger.error(f"Failed to setup quantization: {e}")
            raise
    
    def _log_parameter_info(self) -> None:
        """Log parameter information for PEFT model."""
        if not self.is_peft_enabled:
            return
        
        # Count parameters
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        
        # Calculate percentages
        trainable_percentage = (trainable_params / total_params) * 100 if total_params > 0 else 0
        
        logger.info(f"PEFT Model Parameters:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Trainable percentage: {trainable_percentage:.2f}%")
        
        # Log LoRA-specific info
        if hasattr(self.peft_model, 'peft_config'):
            peft_config = self.peft_model.peft_config
            if peft_config:
                config_key = list(peft_config.keys())[0]  # Get first config
                config = peft_config[config_key]
                logger.info(f"  LoRA rank (r): {config.r}")
                logger.info(f"  LoRA alpha: {config.lora_alpha}")
                logger.info(f"  Target modules: {config.target_modules}")
    
    def enable_adapters(self) -> None:
        """Enable adapter training."""
        if not self.is_peft_enabled:
            logger.warning("PEFT not enabled, cannot enable adapters")
            return
        
        if hasattr(self.peft_model, 'enable_adapter_layers'):
            self.peft_model.enable_adapter_layers()
            logger.info("Adapter layers enabled")
        else:
            logger.warning("Model does not support adapter layer control")
    
    def disable_adapters(self) -> None:
        """Disable adapters for inference."""
        if not self.is_peft_enabled:
            logger.warning("PEFT not enabled, cannot disable adapters")
            return
        
        if hasattr(self.peft_model, 'disable_adapter_layers'):
            self.peft_model.disable_adapter_layers()
            logger.info("Adapter layers disabled")
        else:
            logger.warning("Model does not support adapter layer control")
    
    def get_model(self) -> nn.Module:
        """Get the model (PEFT model if enabled, base model otherwise).
        
        Returns:
            The model to use for training/inference
        """
        if self.is_peft_enabled and self.peft_model is not None:
            return self.peft_model
        return self.base_model
    
    def get_base_model(self) -> nn.Module:
        """Get the base model (without PEFT adapters).
        
        Returns:
            The base model
        """
        return self.base_model
    
    def is_quantized(self) -> bool:
        """Check if model is quantized.
        
        Returns:
            True if model is quantized, False otherwise
        """
        return self.quantization_config is not None
    
    def get_peft_state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get PEFT adapter state dict.
        
        Returns:
            PEFT adapter state dict or None if PEFT not enabled
        """
        if not self.is_peft_enabled or self.peft_model is None:
            return None
        
        try:
            return self.peft_model.state_dict()
        except Exception as e:
            logger.error(f"Failed to get PEFT state dict: {e}")
            return None
    
    def save_adapter(self, save_directory: str) -> bool:
        """Save PEFT adapters to directory.
        
        Args:
            save_directory: Directory to save adapters
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_peft_enabled or self.peft_model is None:
            logger.error("PEFT not enabled, cannot save adapters")
            return False
        
        try:
            save_path = Path(save_directory)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save PEFT adapters
            self.peft_model.save_pretrained(str(save_path))
            
            logger.info(f"PEFT adapters saved to: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save PEFT adapters: {e}")
            return False
    
    def load_adapter(self, adapter_path: str) -> bool:
        """Load PEFT adapters from path.
        
        Args:
            adapter_path: Path to adapter directory
            
        Returns:
            True if successful, False otherwise
        """
        if not PEFT_AVAILABLE:
            logger.error("PEFT not available, cannot load adapters")
            return False
        
        try:
            adapter_path = Path(adapter_path)
            if not adapter_path.exists():
                logger.error(f"Adapter path does not exist: {adapter_path}")
                return False
            
            # Load PEFT model with adapters
            self.peft_model = PeftModel.from_pretrained(self.base_model, str(adapter_path))
            self.is_peft_enabled = True
            
            logger.info(f"PEFT adapters loaded from: {adapter_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load PEFT adapters: {e}")
            return False
    
    def merge_and_unload(self) -> nn.Module:
        """Merge adapters into base model and return merged model.
        
        Returns:
            Merged model with adapters integrated
        """
        if not self.is_peft_enabled or self.peft_model is None:
            logger.warning("PEFT not enabled, returning base model")
            return self.base_model
        
        try:
            # Merge adapters into base model
            merged_model = self.peft_model.merge_and_unload()
            logger.info("Adapters merged into base model")
            return merged_model
            
        except Exception as e:
            logger.error(f"Failed to merge adapters: {e}")
            return self.base_model
    
    def get_memory_footprint(self) -> Dict[str, int]:
        """Get memory footprint information.
        
        Returns:
            Dictionary with memory usage information
        """
        model = self.get_model()
        
        # Calculate parameter memory
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory usage (assuming float32)
        bytes_per_param = 4  # float32
        if self.is_quantized():
            bytes_per_param = 1  # 4-bit quantization
        
        total_memory = total_params * bytes_per_param
        trainable_memory = trainable_params * bytes_per_param
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "total_memory_bytes": total_memory,
            "trainable_memory_bytes": trainable_memory,
            "total_memory_mb": total_memory / (1024 * 1024),
            "trainable_memory_mb": trainable_memory / (1024 * 1024),
            "is_quantized": self.is_quantized()
        }
    
    def print_trainable_parameters(self) -> None:
        """Print detailed information about trainable parameters."""
        model = self.get_model()
        
        print("Trainable Parameters Summary")
        print("=" * 50)
        
        total_params = 0
        trainable_params = 0
        
        for name, param in model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            
            if param.requires_grad:
                trainable_params += param_count
                status = "✓ Trainable"
            else:
                status = "✗ Frozen"
            
            print(f"{name:50} {param_count:>10,} {status}")
        
        print("=" * 50)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        if total_params > 0:
            percentage = (trainable_params / total_params) * 100
            print(f"Trainable percentage: {percentage:.2f}%")
        
        # Memory footprint
        memory_info = self.get_memory_footprint()
        print(f"Estimated memory usage: {memory_info['total_memory_mb']:.1f} MB")
        print(f"Trainable memory usage: {memory_info['trainable_memory_mb']:.1f} MB")


def create_peft_model(base_model: nn.Module, config: Dict[str, Any]) -> PEFTModelWrapper:
    """Create a PEFT model wrapper from configuration.
    
    Args:
        base_model: Base PyTorch model
        config: Full configuration dictionary
        
    Returns:
        PEFTModelWrapper instance
    """
    # Create PEFT configuration manager
    peft_config_manager = PEFTConfigManager(config)
    
    # Create and return wrapper
    return PEFTModelWrapper(base_model, peft_config_manager)


def load_peft_model_from_checkpoint(base_model: nn.Module, checkpoint_path: str, 
                                   config: Dict[str, Any]) -> PEFTModelWrapper:
    """Load PEFT model from checkpoint.
    
    Args:
        base_model: Base PyTorch model
        checkpoint_path: Path to PEFT checkpoint
        config: Full configuration dictionary
        
    Returns:
        PEFTModelWrapper instance with loaded adapters
    """
    # Create wrapper
    wrapper = create_peft_model(base_model, config)
    
    # Load adapters
    if not wrapper.load_adapter(checkpoint_path):
        raise ValueError(f"Failed to load PEFT adapters from: {checkpoint_path}")
    
    return wrapper


def estimate_peft_memory_savings(base_model: nn.Module, config: Dict[str, Any]) -> Dict[str, float]:
    """Estimate memory savings from using PEFT.
    
    Args:
        base_model: Base PyTorch model
        config: Full configuration dictionary
        
    Returns:
        Dictionary with memory savings estimates
    """
    # Create temporary wrapper to get estimates
    peft_config_manager = PEFTConfigManager(config)
    
    if not peft_config_manager.is_enabled():
        return {"memory_savings_percentage": 0.0, "memory_reduction_factor": 1.0}
    
    # Calculate base model memory
    total_params = sum(p.numel() for p in base_model.parameters())
    base_memory = total_params * 4  # float32 bytes
    
    # Estimate PEFT memory
    validator = PEFTTargetModuleValidator(base_model, peft_config_manager)
    param_estimates = validator.estimate_lora_parameters()
    
    peft_memory = param_estimates["total_lora_parameters"] * 4  # LoRA params
    
    # Add quantization savings if using QLoRA
    if peft_config_manager.is_qlora_enabled():
        base_memory = total_params * 1  # 4-bit quantization
    
    total_peft_memory = base_memory + peft_memory
    
    # Calculate savings
    memory_savings = (base_memory - total_peft_memory) / base_memory * 100
    memory_reduction_factor = base_memory / total_peft_memory
    
    return {
        "base_memory_mb": base_memory / (1024 * 1024),
        "peft_memory_mb": total_peft_memory / (1024 * 1024),
        "memory_savings_percentage": max(0, memory_savings),
        "memory_reduction_factor": memory_reduction_factor,
        "lora_parameters": param_estimates["total_lora_parameters"],
        "lora_percentage": param_estimates["lora_percentage"]
    }