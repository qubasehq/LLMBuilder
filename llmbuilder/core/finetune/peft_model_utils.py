"""
PEFT Model Utilities

This module provides utilities for working with PEFT models, including target module
validation and model architecture inspection.
"""

import logging
from typing import List, Set, Dict, Any, Optional
from llmbuilder.utils.lazy_imports import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelArchitectureInspector:
    """Inspects model architecture to validate PEFT target modules."""
    
    def __init__(self, model: nn.Module):
        """Initialize with a PyTorch model.
        
        Args:
            model: PyTorch model to inspect
        """
        self.model = model
        self._module_names = None
        self._linear_modules = None
    
    def get_all_module_names(self) -> Set[str]:
        """Get all module names in the model.
        
        Returns:
            Set of all module names
        """
        if self._module_names is None:
            self._module_names = set()
            for name, _ in self.model.named_modules():
                if name:  # Skip empty name (root module)
                    self._module_names.add(name)
        return self._module_names
    
    def get_linear_module_names(self) -> Set[str]:
        """Get names of all linear/dense modules in the model.
        
        Returns:
            Set of linear module names
        """
        if self._linear_modules is None:
            self._linear_modules = set()
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d)) and name:
                    self._linear_modules.add(name)
        return self._linear_modules
    
    def validate_target_modules(self, target_modules: List[str]) -> Dict[str, bool]:
        """Validate that target modules exist in the model.
        
        Args:
            target_modules: List of target module names to validate
            
        Returns:
            Dictionary mapping module names to validation status
        """
        all_modules = self.get_all_module_names()
        validation_results = {}
        
        for module_name in target_modules:
            # Check exact match
            if module_name in all_modules:
                validation_results[module_name] = True
                continue
            
            # Check if it's a pattern that matches multiple modules
            matching_modules = [name for name in all_modules if module_name in name]
            if matching_modules:
                validation_results[module_name] = True
                logger.info(f"Target module '{module_name}' matches: {matching_modules}")
            else:
                validation_results[module_name] = False
                logger.warning(f"Target module '{module_name}' not found in model")
        
        return validation_results
    
    def suggest_target_modules(self, architecture_type: str = "gpt") -> List[str]:
        """Suggest appropriate target modules based on model architecture.
        
        Args:
            architecture_type: Type of model architecture
            
        Returns:
            List of suggested target module names
        """
        linear_modules = self.get_linear_module_names()
        
        if architecture_type.lower() == "gpt":
            # Common GPT-style module patterns
            suggested = []
            
            # Look for attention modules
            attention_patterns = ["attn", "attention", "self_attn"]
            for pattern in attention_patterns:
                matching = [name for name in linear_modules if pattern in name.lower()]
                suggested.extend(matching)
            
            # Look for MLP/feed-forward modules
            mlp_patterns = ["mlp", "ffn", "feed_forward", "fc"]
            for pattern in mlp_patterns:
                matching = [name for name in linear_modules if pattern in name.lower()]
                suggested.extend(matching)
            
            # Remove duplicates while preserving order
            suggested = list(dict.fromkeys(suggested))
            
            if suggested:
                logger.info(f"Suggested target modules for {architecture_type}: {suggested}")
                return suggested
        
        # Fallback: return all linear modules
        logger.info(f"Using all linear modules as targets: {list(linear_modules)}")
        return list(linear_modules)
    
    def get_module_info(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific module.
        
        Args:
            module_name: Name of the module to inspect
            
        Returns:
            Dictionary with module information or None if not found
        """
        try:
            module = dict(self.model.named_modules())[module_name]
            
            info = {
                "name": module_name,
                "type": type(module).__name__,
                "parameters": sum(p.numel() for p in module.parameters()),
                "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad)
            }
            
            # Add specific info for linear modules
            if isinstance(module, nn.Linear):
                info.update({
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "bias": module.bias is not None
                })
            
            return info
            
        except KeyError:
            logger.error(f"Module '{module_name}' not found in model")
            return None
    
    def print_model_structure(self, max_depth: int = 3) -> None:
        """Print model structure for debugging.
        
        Args:
            max_depth: Maximum depth to print
        """
        print("Model Structure:")
        print("=" * 50)
        
        for name, module in self.model.named_modules():
            if not name:  # Skip root module
                continue
            
            depth = name.count('.')
            if depth > max_depth:
                continue
            
            indent = "  " * depth
            module_type = type(module).__name__
            
            # Add parameter count for leaf modules
            if not list(module.children()):
                param_count = sum(p.numel() for p in module.parameters())
                print(f"{indent}{name}: {module_type} ({param_count:,} params)")
            else:
                print(f"{indent}{name}: {module_type}")


class PEFTTargetModuleValidator:
    """Validates and manages PEFT target modules."""
    
    def __init__(self, model: nn.Module, config_manager):
        """Initialize validator.
        
        Args:
            model: PyTorch model
            config_manager: PEFT configuration manager
        """
        self.model = model
        self.config_manager = config_manager
        self.inspector = ModelArchitectureInspector(model)
    
    def validate_and_fix_target_modules(self) -> List[str]:
        """Validate target modules and suggest fixes if needed.
        
        Returns:
            List of validated target module names
        """
        target_modules = self.config_manager.get_target_modules()
        
        # Validate current target modules
        validation_results = self.inspector.validate_target_modules(target_modules)
        
        # Check if all modules are valid
        invalid_modules = [name for name, valid in validation_results.items() if not valid]
        
        if invalid_modules:
            logger.warning(f"Invalid target modules found: {invalid_modules}")
            
            # Suggest alternative modules
            suggested_modules = self.inspector.suggest_target_modules()
            
            if suggested_modules:
                logger.info(f"Suggested alternative target modules: {suggested_modules}")
                
                # Filter to keep only valid modules from original list
                valid_modules = [name for name, valid in validation_results.items() if valid]
                
                # Add suggested modules that aren't already included
                for suggested in suggested_modules:
                    if suggested not in valid_modules:
                        valid_modules.append(suggested)
                
                return valid_modules
            else:
                raise ValueError(
                    f"No valid target modules found. Invalid modules: {invalid_modules}. "
                    f"Available linear modules: {list(self.inspector.get_linear_module_names())}"
                )
        
        return target_modules
    
    def get_target_module_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about target modules.
        
        Returns:
            Dictionary mapping module names to their information
        """
        target_modules = self.validate_and_fix_target_modules()
        module_info = {}
        
        for module_name in target_modules:
            info = self.inspector.get_module_info(module_name)
            if info:
                module_info[module_name] = info
        
        return module_info
    
    def estimate_lora_parameters(self) -> Dict[str, int]:
        """Estimate the number of parameters that will be added by LoRA.
        
        Returns:
            Dictionary with parameter count estimates
        """
        target_modules = self.validate_and_fix_target_modules()
        lora_config = self.config_manager.lora_config
        
        total_lora_params = 0
        module_params = {}
        
        for module_name in target_modules:
            info = self.inspector.get_module_info(module_name)
            if info and info["type"] == "Linear":
                # LoRA adds two matrices: A (in_features x r) and B (r x out_features)
                in_features = info["in_features"]
                out_features = info["out_features"]
                r = lora_config.r
                
                # Parameters for this module: A matrix + B matrix
                module_lora_params = (in_features * r) + (r * out_features)
                module_params[module_name] = module_lora_params
                total_lora_params += module_lora_params
        
        # Get total model parameters for comparison
        total_model_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total_model_parameters": total_model_params,
            "trainable_model_parameters": trainable_params,
            "total_lora_parameters": total_lora_params,
            "lora_percentage": (total_lora_params / total_model_params) * 100,
            "module_breakdown": module_params
        }
    
    def print_target_module_summary(self) -> None:
        """Print a summary of target modules and LoRA parameter estimates."""
        print("PEFT Target Module Summary")
        print("=" * 50)
        
        # Validate modules
        target_modules = self.validate_and_fix_target_modules()
        print(f"Target modules ({len(target_modules)}):")
        for module in target_modules:
            print(f"  - {module}")
        
        print()
        
        # Parameter estimates
        param_estimates = self.estimate_lora_parameters()
        print("Parameter Estimates:")
        print(f"  Total model parameters: {param_estimates['total_model_parameters']:,}")
        print(f"  LoRA parameters: {param_estimates['total_lora_parameters']:,}")
        print(f"  LoRA percentage: {param_estimates['lora_percentage']:.2f}%")
        
        print()
        print("Per-module LoRA parameters:")
        for module, params in param_estimates['module_breakdown'].items():
            print(f"  {module}: {params:,}")


def validate_model_for_peft(model: nn.Module, config_manager) -> bool:
    """Validate that a model is compatible with PEFT configuration.
    
    Args:
        model: PyTorch model to validate
        config_manager: PEFT configuration manager
        
    Returns:
        True if model is compatible, False otherwise
    """
    try:
        validator = PEFTTargetModuleValidator(model, config_manager)
        target_modules = validator.validate_and_fix_target_modules()
        
        if not target_modules:
            logger.error("No valid target modules found for PEFT")
            return False
        
        # Check if model has trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable_params == 0:
            logger.error("Model has no trainable parameters")
            return False
        
        logger.info(f"Model validation successful. Target modules: {target_modules}")
        return True
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return False


def get_recommended_lora_rank(model: nn.Module, target_modules: List[str]) -> int:
    """Get recommended LoRA rank based on model size and target modules.
    
    Args:
        model: PyTorch model
        target_modules: List of target module names
        
    Returns:
        Recommended LoRA rank
    """
    inspector = ModelArchitectureInspector(model)
    
    # Calculate average dimension of target modules
    total_dims = 0
    module_count = 0
    
    for module_name in target_modules:
        info = inspector.get_module_info(module_name)
        if info and info["type"] == "Linear":
            # Use the smaller dimension as a reference
            min_dim = min(info["in_features"], info["out_features"])
            total_dims += min_dim
            module_count += 1
    
    if module_count == 0:
        return 16  # Default rank
    
    avg_dim = total_dims / module_count
    
    # Recommend rank based on average dimension
    if avg_dim < 256:
        return 8
    elif avg_dim < 512:
        return 16
    elif avg_dim < 1024:
        return 32
    elif avg_dim < 2048:
        return 64
    else:
        return 128