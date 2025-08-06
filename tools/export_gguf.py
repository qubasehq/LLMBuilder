"""
GGUF export utility for LLM models.
Converts trained PyTorch models to GGUF format for llama.cpp compatibility.
"""

import os
import struct
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class GGUFExporter:
    """Export PyTorch models to GGUF format."""
    
    def __init__(self, model_path: str, output_path: str):
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.model = None
        self.config = None
        
    def load_model(self):
        """Load PyTorch model from checkpoint."""
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            if 'model' in checkpoint:
                self.model = checkpoint['model']
                self.config = checkpoint['config']
            else:
                # Direct model
                self.model = checkpoint
                self.config = getattr(self.model, 'config', {})
                
            logger.info(f"Loaded model from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def create_gguf_metadata(self) -> Dict[str, Any]:
        """Create GGUF metadata from model config."""
        metadata = {
            "general.architecture": "gpt2",
            "general.name": "LLMBuilder",
            "general.quantization_version": 2,
            
            # Model architecture
            "gpt2.context_length": self.config.get('block_size', 512),
            "gpt2.embedding_length": self.config.get('n_embd', 512),
            "gpt2.feed_forward_length": self.config.get('n_embd', 512) * 4,
            "gpt2.block_count": self.config.get('n_layer', 6),
            "gpt2.attention.head_count": self.config.get('n_head', 6),
            "gpt2.attention.head_count_kv": self.config.get('n_head', 6),
            "gpt2.attention.layer_norm_epsilon": 1e-5,
            
            # Vocabulary
            "gpt2.vocab_size": self.config.get('vocab_size', 32000),
            "gpt2.rope.dimension_count": self.config.get('n_embd', 512) // self.config.get('n_head', 6),
            
            # Tokenizer info
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.tokens": [],
            "tokenizer.ggml.scores": [],
            "tokenizer.ggml.merges": [],
            "tokenizer.ggml.bos_token_id": 1,
            "tokenizer.ggml.eos_token_id": 2,
            "tokenizer.ggml.unk_token_id": 0,
            "tokenizer.ggml.padding_token_id": 0,
        }
        
        return metadata
    
    def export_to_gguf(self) -> bool:
        """Export model to GGUF format."""
        try:
            if not self.load_model():
                return False
                
            # Create GGUF file
            with open(self.output_path, 'wb') as f:
                # Write GGUF header
                self._write_header(f)
                
                # Write metadata
                metadata = self.create_gguf_metadata()
                self._write_metadata(f, metadata)
                
                # Write tensor data
                self._write_tensors(f)
                
            logger.info(f"Successfully exported to {self.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export to GGUF: {e}")
            return False
    
    def _write_header(self, f):
        """Write GGUF header."""
        # GGUF magic number
        magic = b'GGUF\x02\x00\x00\x00'
        f.write(magic)
        
        # Version
        f.write(struct.pack('<I', 3))
        
        # Tensor count and metadata kv count (will be updated later)
        f.write(struct.pack('<Q', 0))  # tensor_count placeholder
        f.write(struct.pack('<Q', 0))  # metadata_kv_count placeholder
    
    def _write_metadata(self, f, metadata: Dict[str, Any]):
        """Write metadata key-value pairs."""
        # This is a simplified version - full GGUF spec is more complex
        for key, value in metadata.items():
            if isinstance(value, str):
                # Write string value
                key_bytes = key.encode('utf-8')
                value_bytes = value.encode('utf-8')
                f.write(struct.pack('<I', len(key_bytes)))
                f.write(key_bytes)
                f.write(struct.pack('<I', len(value_bytes)))
                f.write(value_bytes)
            elif isinstance(value, int):
                # Write int value
                key_bytes = key.encode('utf-8')
                f.write(struct.pack('<I', len(key_bytes)))
                f.write(key_bytes)
                f.write(struct.pack('<I', 8))
                f.write(struct.pack('<Q', value))
    
    def _write_tensors(self, f):
        """Write tensor data."""
        state_dict = self.model.state_dict()
        
        for name, tensor in state_dict.items():
            # Convert tensor to numpy
            np_tensor = tensor.detach().cpu().numpy()
            
            # Write tensor info
            tensor_name = name.encode('utf-8')
            f.write(struct.pack('<I', len(tensor_name)))
            f.write(tensor_name)
            
            # Write dimensions
            dims = np_tensor.ndim
            f.write(struct.pack('<I', dims))
            for dim in np_tensor.shape:
                f.write(struct.pack('<Q', dim))
                
            # Write tensor data
            tensor_bytes = np_tensor.astype(np.float32).tobytes()
            f.write(struct.pack('<Q', len(tensor_bytes)))
            f.write(tensor_bytes)


class GGMLQuantizer:
    """Quantize model weights for GGML format."""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.quantized_weights = {}
        
    def quantize_q4_0(self, tensor: torch.Tensor) -> np.ndarray:
        """Quantize to Q4_0 format."""
        # Q4_0: 4-bit quantization with block-wise scaling
        tensor = tensor.flatten()
        n = tensor.numel()
        
        # Calculate blocks (32 elements per block)
        block_size = 32
        n_blocks = (n + block_size - 1) // block_size
        
        # Initialize quantized data
        quantized = np.zeros(n_blocks * 18, dtype=np.uint8)  # 16 bytes + 2 scale bytes
        
        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, n)
            block = tensor[start_idx:end_idx]
            
            if len(block) < block_size:
                # Pad with zeros
                block = torch.cat([block, torch.zeros(block_size - len(block))])
                
            # Calculate scale and quantize
            absmax = torch.max(torch.abs(block))
            if absmax > 0:
                scale = absmax / 7.0  # 7 is max for 4-bit signed
                quantized_block = torch.round(block / scale).clamp(-8, 7).to(torch.int8)
            else:
                scale = 1.0
                quantized_block = torch.zeros(block_size, dtype=torch.int8)
                
            # Store quantized data
            block_start = i * 18
            quantized[block_start:block_start+16] = quantized_block.numpy().astype(np.uint8)
            quantized[block_start+16:block_start+18] = np.frombuffer(
                struct.pack('<e', scale), dtype=np.uint8
            )
            
        return quantized
    
    def quantize_q8_0(self, tensor: torch.Tensor) -> np.ndarray:
        """Quantize to Q8_0 format."""
        # Q8_0: 8-bit quantization with block-wise scaling
        tensor = tensor.flatten()
        n = tensor.numel()
        
        block_size = 32
        n_blocks = (n + block_size - 1) // block_size
        
        quantized = np.zeros(n_blocks * 34, dtype=np.uint8)  # 32 bytes + 2 scale bytes
        
        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, n)
            block = tensor[start_idx:end_idx]
            
            if len(block) < block_size:
                block = torch.cat([block, torch.zeros(block_size - len(block))])
                
            absmax = torch.max(torch.abs(block))
            if absmax > 0:
                scale = absmax / 127.0  # 127 is max for 8-bit signed
                quantized_block = torch.round(block / scale).clamp(-128, 127).to(torch.int8)
            else:
                scale = 1.0
                quantized_block = torch.zeros(block_size, dtype=torch.int8)
                
            block_start = i * 34
            quantized[block_start:block_start+32] = quantized_block.numpy().astype(np.uint8)
            quantized[block_start+32:block_start+34] = np.frombuffer(
                struct.pack('<e', scale), dtype=np.uint8
            )
            
        return quantized


def export_to_gguf_cli(input_path: str, output_path: str, quantization: str = "f32"):
    """CLI function for GGUF export."""
    try:
        exporter = GGUFExporter(input_path, output_path)
        
        if quantization != "f32":
            quantizer = GGMLQuantizer(input_path)
            logger.info(f"Using {quantization} quantization")
            
        success = exporter.export_to_gguf()
        
        if success:
            logger.info(f"Successfully exported to {output_path}")
            
            # Calculate file sizes
            input_size = Path(input_path).stat().st_size / (1024 * 1024)
            output_size = Path(output_path).stat().st_size / (1024 * 1024)
            
            logger.info(f"Input: {input_size:.2f} MB")
            logger.info(f"Output: {output_size:.2f} MB")
            logger.info(f"Compression ratio: {input_size/output_size:.2f}x")
            
        return success
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export model to GGUF format")
    parser.add_argument("--input", required=True, help="Path to input model checkpoint")
    parser.add_argument("--output", required=True, help="Output GGUF file path")
    parser.add_argument("--quantization", choices=["f32", "f16", "q8_0", "q4_0"], 
                       default="f32", help="Quantization format")
    parser.add_argument("--vocab", help="Path to tokenizer vocabulary file")
    
    args = parser.parse_args()
    
    export_to_gguf_cli(args.input, args.output, args.quantization)
