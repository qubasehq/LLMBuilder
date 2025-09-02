"""
Quantization utilities for LLM training and inference.
Supports various quantization schemes for model compression.
"""

from llmbuilder.utils.lazy_imports import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
from llmbuilder.utils.lazy_imports import numpy as np
from loguru import logger


class QuantConfig:
    """Configuration for quantization parameters."""
    
    def __init__(
        self,
        bits: int = 8,
        scheme: str = "symmetric",  # "symmetric", "asymmetric", "dynamic"
        granularity: str = "per_tensor",  # "per_tensor", "per_channel", "per_group"
        group_size: int = 128,
        enable_zero_point: bool = True,
        enable_scale: bool = True,
        dtype: torch.dtype = torch.int8
    ):
        self.bits = bits
        self.scheme = scheme
        self.granularity = granularity
        self.group_size = group_size
        self.enable_zero_point = enable_zero_point
        self.enable_scale = enable_scale
        self.dtype = dtype
        
        # Validate parameters
        if bits not in [4, 8, 16]:
            raise ValueError(f"Unsupported bits: {bits}. Use 4, 8, or 16.")
        
        if scheme not in ["symmetric", "asymmetric", "dynamic"]:
            raise ValueError(f"Unsupported scheme: {scheme}")
            
        if granularity not in ["per_tensor", "per_channel", "per_group"]:
            raise ValueError(f"Unsupported granularity: {granularity}")


class QuantizedLinear(nn.Module):
    """Quantized linear layer implementation."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: QuantConfig = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantConfig()
        
        # Initialize full precision weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Quantized parameters
        self.register_buffer('quant_weight', None)
        self.register_buffer('weight_scale', None)
        self.register_buffer('weight_zero_point', None)
        
        # Quantization state
        self.is_quantized = False
        
    def quantize_weights(self, calibration_data: Optional[torch.Tensor] = None):
        """Quantize the weights using specified configuration."""
        if self.is_quantized:
            logger.warning("Weights already quantized")
            return
            
        weight = self.weight.data
        
        if self.config.scheme == "symmetric":
            # Symmetric quantization
            max_val = torch.max(torch.abs(weight))
            scale = max_val / (2**(self.config.bits - 1) - 1)
            zero_point = 0
            
            # Quantize
            quant_weight = torch.round(weight / scale).clamp(
                -(2**(self.config.bits - 1)), 2**(self.config.bits - 1) - 1
            ).to(self.config.dtype)
            
        elif self.config.scheme == "asymmetric":
            # Asymmetric quantization
            min_val = torch.min(weight)
            max_val = torch.max(weight)
            scale = (max_val - min_val) / (2**self.config.bits - 1)
            zero_point = torch.round(-min_val / scale)
            
            # Quantize
            quant_weight = torch.round((weight / scale) + zero_point).clamp(
                0, 2**self.config.bits - 1
            ).to(self.config.dtype)
            
        else:  # dynamic
            # Dynamic quantization (will be handled at runtime)
            quant_weight = weight
            scale = torch.tensor(1.0)
            zero_point = torch.tensor(0.0)
            
        # Store quantized parameters
        self.register_buffer('quant_weight', quant_weight)
        self.register_buffer('weight_scale', scale)
        self.register_buffer('weight_zero_point', zero_point)
        
        self.is_quantized = True
        logger.info(f"Quantized linear layer: {self.in_features}x{self.out_features}")
        
    def dequantize_weights(self):
        """Dequantize weights back to full precision."""
        if not self.is_quantized:
            return self.weight
            
        if self.config.scheme == "symmetric":
            return self.quant_weight.float() * self.weight_scale
        elif self.config.scheme == "asymmetric":
            return (self.quant_weight.float() - self.weight_zero_point) * self.weight_scale
        else:
            return self.quant_weight.float()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights."""
        if self.is_quantized:
            weight = self.dequantize_weights()
        else:
            weight = self.weight
            
        output = torch.nn.functional.linear(x, weight, self.bias)
        return output


class QuantizedGPTModel(nn.Module):
    """Quantized version of GPT model."""
    
    def __init__(self, config: Dict[str, Any], quant_config: QuantConfig = None):
        super().__init__()
        self.config = config
        self.quant_config = quant_config or QuantConfig()
        
        # Create quantized layers
        self.token_embedding = nn.Embedding(config['vocab_size'], config['n_embd'])
        self.position_embedding = nn.Embedding(config['block_size'], config['n_embd'])
        
        # Quantized transformer blocks
        self.blocks = nn.ModuleList([
            QuantizedTransformerBlock(config, self.quant_config)
            for _ in range(config['n_layer'])
        ])
        
        self.ln_f = nn.LayerNorm(config['n_embd'])
        self.lm_head = QuantizedLinear(
            config['n_embd'], 
            config['vocab_size'], 
            bias=False,
            config=self.quant_config
        )
        
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def quantize_model(self):
        """Quantize the entire model."""
        logger.info("Starting model quantization...")
        
        # Quantize transformer blocks
        for block in self.blocks:
            block.quantize()
            
        # Quantize final layers
        if hasattr(self.lm_head, 'quantize_weights'):
            self.lm_head.quantize_weights()
            
        logger.info("Model quantization completed")
        
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """Forward pass."""
        device = idx.device
        b, t = idx.size()
        
        # Token and position embeddings
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(t, device=device))
        x = tok_emb + pos_emb
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-1
            )
            return logits, loss
        else:
            return logits, None


class QuantizedTransformerBlock(nn.Module):
    """Quantized transformer block."""
    
    def __init__(self, config: Dict[str, Any], quant_config: QuantConfig):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        
        # Quantized layers
        self.ln_1 = nn.LayerNorm(config['n_embd'])
        self.attn = QuantizedMultiHeadAttention(config, quant_config)
        self.ln_2 = nn.LayerNorm(config['n_embd'])
        self.mlp = QuantizedMLP(config, quant_config)
        
    def quantize(self):
        """Quantize this block."""
        self.attn.quantize()
        self.mlp.quantize()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Self-attention with residual connection
        x = x + self.attn(self.ln_1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.ln_2(x))
        
        return x


class QuantizedMultiHeadAttention(nn.Module):
    """Quantized multi-head attention."""
    
    def __init__(self, config: Dict[str, Any], quant_config: QuantConfig):
        super().__init__()
        self.config = config
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        self.head_size = self.n_embd // self.n_head
        
        # Quantized linear layers
        self.c_attn = QuantizedLinear(
            self.n_embd, 
            3 * self.n_embd, 
            config=quant_config
        )
        self.c_proj = QuantizedLinear(
            self.n_embd, 
            self.n_embd, 
            config=quant_config
        )
        
        # Causal mask
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(config['block_size'], config['block_size']))
            .view(1, 1, config['block_size'], config['block_size'])
        )
        
    def quantize(self):
        """Quantize attention layers."""
        if hasattr(self.c_attn, 'quantize_weights'):
            self.c_attn.quantize_weights()
        if hasattr(self.c_proj, 'quantize_weights'):
            self.c_proj.quantize_weights()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        B, T, C = x.size()
        
        # Calculate query, key, values
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        y = att @ v
        
        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        
        return y


class QuantizedMLP(nn.Module):
    """Quantized MLP block."""
    
    def __init__(self, config: Dict[str, Any], quant_config: QuantConfig):
        super().__init__()
        self.config = config
        
        # Quantized layers
        self.c_fc = QuantizedLinear(
            config['n_embd'], 
            4 * config['n_embd'], 
            config=quant_config
        )
        self.c_proj = QuantizedLinear(
            4 * config['n_embd'], 
            config['n_embd'], 
            config=quant_config
        )
        
    def quantize(self):
        """Quantize MLP layers."""
        if hasattr(self.c_fc, 'quantize_weights'):
            self.c_fc.quantize_weights()
        if hasattr(self.c_proj, 'quantize_weights'):
            self.c_proj.quantize_weights()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.c_fc(x)
        x = torch.nn.functional.gelu(x)
        x = self.c_proj(x)
        return x


class QuantizationManager:
    """Manager for model quantization operations."""
    
    @staticmethod
    def quantize_model(model: nn.Module, config: QuantConfig) -> nn.Module:
        """Quantize a full model."""
        if isinstance(model, QuantizedGPTModel):
            model.quantize_model()
            return model
            
        # Convert regular model to quantized
        from llmbuilder.core.model.gpt_model import GPTModel
        if isinstance(model, GPTModel):
            # Create quantized version
            quantized_model = QuantizedGPTModel(
                model.config, 
                config
            )
            
            # Copy weights
            quantized_model.load_state_dict(model.state_dict())
            quantized_model.quantize_model()
            
            return quantized_model
            
        raise ValueError("Unsupported model type for quantization")
    
    @staticmethod
    def get_model_size(model: nn.Module) -> Dict[str, float]:
        """Get model size information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate size in MB
        param_size = total_params * 4 / (1024 * 1024)  # float32
        buffer_size = sum(b.numel() for b in model.buffers()) * 4 / (1024 * 1024)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': param_size + buffer_size
        }
    
    @staticmethod
    def estimate_quantized_size(original_size_mb: float, bits: int) -> float:
        """Estimate size after quantization."""
        compression_ratio = 32 / bits  # from float32
        return original_size_mb / compression_ratio


# Import math for transformer blocks
import math


def quantize_model_cli(model_path: str, output_path: str, bits: int = 8):
    """CLI function for model quantization."""
    import argparse
    
    config = QuantConfig(bits=bits)
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create quantized model
    quantized_model = QuantizationManager.quantize_model(
        checkpoint['model'], 
        config
    )
    
    # Save quantized model
    torch.save({
        'model': quantized_model,
        'config': checkpoint['config'],
        'quant_config': config,
        'metrics': checkpoint.get('metrics', {})
    }, output_path)
    
    logger.info(f"Quantized model saved to {output_path}")


if __name__ == "__main__":
    # CLI interface
    parser = argparse.ArgumentParser(description="Quantize LLM model")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", required=True, help="Output path for quantized model")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8, 16], help="Quantization bits")
    parser.add_argument("--scheme", default="symmetric", choices=["symmetric", "asymmetric"], help="Quantization scheme")
    
    args = parser.parse_args()
    
    config = QuantConfig(bits=args.bits, scheme=args.scheme)
    
    quantize_model_cli(args.model, args.output, args.bits)
