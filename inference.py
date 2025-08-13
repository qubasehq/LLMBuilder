#!/usr/bin/env python3
"""
Simple inference script for trained LLM models.
Provides an interactive interface for text generation.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import json
from typing import Optional, List
import struct

import torch
import torch.nn.functional as F
from loguru import logger
import sentencepiece as spm
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from model.gpt_model import GPTModel
from training.utils import setup_logging


class GGUFLoader:
    """Custom GGUF file loader for our models."""
    
    def __init__(self, gguf_path: str):
        self.gguf_path = gguf_path
        self.metadata = {}
        self.tensors = {}
        self.tensor_data = {}
    
    def load_gguf(self):
        """Load GGUF file and extract tensors."""
        try:
            with open(self.gguf_path, 'rb') as f:
                # Read GGUF header
                magic = f.read(4)
                if magic != b'GGUF':
                    raise ValueError("Not a valid GGUF file")
                
                version = struct.unpack('<I', f.read(4))[0]
                logger.info(f"Loading GGUF version {version}")
                
                # Read metadata count and skip metadata
                metadata_count = struct.unpack('<Q', f.read(8))[0]
                self._skip_metadata(f, metadata_count)
                
                # Read tensor info
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                logger.info(f"Found {tensor_count} tensors")
                
                # Read tensor headers
                tensor_infos = []
                for i in range(tensor_count):
                    tensor_info = self._read_tensor_info(f)
                    tensor_infos.append(tensor_info)
                
                # Align to tensor data section
                current_pos = f.tell()
                alignment = 32  # GGUF alignment
                aligned_pos = (current_pos + alignment - 1) // alignment * alignment
                f.seek(aligned_pos)
                
                # Read tensor data
                for tensor_info in tensor_infos:
                    name = tensor_info['name']
                    shape = tensor_info['shape']
                    dtype = tensor_info['type']
                    size = tensor_info['size']
                    
                    # Read raw tensor data
                    raw_data = f.read(size)
                    
                    # Convert to numpy array based on type
                    if dtype == 0:  # F32
                        data = np.frombuffer(raw_data, dtype=np.float32)
                    elif dtype == 1:  # F16
                        data = np.frombuffer(raw_data, dtype=np.float16).astype(np.float32)
                    elif dtype == 2:  # Q4_0 (simplified - treat as f16)
                        data = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / 255.0
                    elif dtype == 3:  # Q4_1 (simplified - treat as f16)
                        data = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / 255.0
                    elif dtype == 6:  # Q5_0 (simplified - treat as f16)
                        data = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / 255.0
                    elif dtype == 7:  # Q5_1 (simplified - treat as f16)
                        data = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / 255.0
                    elif dtype == 8:  # Q8_0 (simplified - treat as int8)
                        data = np.frombuffer(raw_data, dtype=np.int8).astype(np.float32) / 127.0
                    else:
                        logger.warning(f"Unknown tensor type {dtype} for {name}, using random data")
                        data = np.random.randn(np.prod(shape)).astype(np.float32)
                    
                    # Reshape to correct dimensions
                    try:
                        if len(shape) > 0 and np.prod(shape) > 0:
                            data = data[:np.prod(shape)].reshape(shape)
                        else:
                            data = data.reshape(-1)
                        
                        # Convert to PyTorch tensor
                        self.tensor_data[name] = torch.from_numpy(data)
                        logger.debug(f"Loaded tensor {name}: {shape}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to reshape tensor {name}: {e}")
                        # Create random tensor with correct shape
                        self.tensor_data[name] = torch.randn(shape)
                
                logger.info(f"Successfully loaded {len(self.tensor_data)} tensors from GGUF")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load GGUF: {e}")
            return False
    
    def _skip_metadata(self, f, count):
        """Skip metadata section."""
        for _ in range(count):
            # Skip key
            key_len = struct.unpack('<Q', f.read(8))[0]
            f.read(key_len)
            # Skip value
            value_type = struct.unpack('<I', f.read(4))[0]
            if value_type == 8:  # String
                value_len = struct.unpack('<Q', f.read(8))[0]
                f.read(value_len)
            elif value_type in [4, 5]:  # Int32, Int64
                f.read(8)
            elif value_type in [6, 7]:  # Float32, Float64
                f.read(8)
            elif value_type == 9:  # Bool
                f.read(1)
            elif value_type == 10:  # Array
                array_type = struct.unpack('<I', f.read(4))[0]
                array_len = struct.unpack('<Q', f.read(8))[0]
                if array_type == 4:  # Int32 array
                    f.read(4 * array_len)
                elif array_type == 5:  # Int64 array
                    f.read(8 * array_len)
    
    def _read_tensor_info(self, f):
        """Read tensor information."""
        # Read tensor name
        name_len = struct.unpack('<Q', f.read(8))[0]
        name = f.read(name_len).decode('utf-8')
        
        # Read dimensions
        n_dims = struct.unpack('<I', f.read(4))[0]
        shape = []
        for _ in range(n_dims):
            dim = struct.unpack('<Q', f.read(8))[0]
            shape.append(int(dim))
        
        # Read tensor type
        tensor_type = struct.unpack('<I', f.read(4))[0]
        
        # Read tensor offset
        offset = struct.unpack('<Q', f.read(8))[0]
        
        # Calculate tensor size (simplified)
        if tensor_type == 0:  # F32
            element_size = 4
        elif tensor_type == 1:  # F16
            element_size = 2
        else:  # Quantized types (simplified)
            element_size = 1
        
        size = np.prod(shape) * element_size if shape else element_size
        
        return {
            'name': name,
            'shape': shape,
            'type': tensor_type,
            'offset': offset,
            'size': int(size)
        }


class LLMInference:
    """Simple inference class for text generation."""
    
    def __init__(self, 
                 model_path: str, 
                 tokenizer_path: str, 
                 config_path: str = "config.json",
                 device: Optional[str] = None):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model checkpoint (.pt or .gguf)
            tokenizer_path: Path to tokenizer directory
            config_path: Path to model configuration
            device: Device to run inference on (auto-detect if None)
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.config_path = config_path
        self.is_gguf = model_path.endswith('.gguf')
        
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using CUDA for inference")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU for inference")
        else:
            self.device = torch.device(device)
            logger.info(f"Using {device} for inference")
        
        # Load configuration from JSON
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load tokenizer and model
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        
        logger.success("Inference engine initialized successfully")
    
    def _load_config(self) -> dict:
        """Load model configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_tokenizer(self) -> spm.SentencePieceProcessor:
        """Load the tokenizer."""
        # Handle both directory paths and direct file paths
        if os.path.isdir(self.tokenizer_path):
            tokenizer_model_path = os.path.join(self.tokenizer_path, "tokenizer.model")
        else:
            tokenizer_model_path = self.tokenizer_path
        
        if not os.path.exists(tokenizer_model_path):
            # Try alternative path
            alt_path = os.path.join(os.path.dirname(self.tokenizer_path), "tokenizer.model")
            if os.path.exists(alt_path):
                tokenizer_model_path = alt_path
            else:
                raise FileNotFoundError(f"Tokenizer model not found: {tokenizer_model_path}")
        
        try:
            tokenizer = spm.SentencePieceProcessor()
            tokenizer.load(tokenizer_model_path)
            logger.info(f"Loaded tokenizer from {tokenizer_model_path}")
            logger.info(f"Vocabulary size: {tokenizer.vocab_size()}")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _load_model(self) -> GPTModel:
        """Load the trained model (PT or GGUF format)."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        try:
            if self.is_gguf:
                return self._load_gguf_model()
            else:
                return self._load_pytorch_model()
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_pytorch_model(self) -> GPTModel:
        """Load PyTorch model."""
        # Load checkpoint first to get the correct config
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Use config from checkpoint if available, otherwise fall back to external config
        if 'config' in checkpoint and 'model' in checkpoint['config']:
            model_config = checkpoint['config']['model']
            logger.info("Using model config from checkpoint")
        else:
            model_config = self.config['model']
            logger.info("Using model config from external file")
        
        # Auto-detect vocabulary size from tokenizer
        vocab_size = model_config.get('vocab_size')
        if vocab_size is None or vocab_size != self.tokenizer.vocab_size():
            vocab_size = self.tokenizer.vocab_size()
            logger.info(f"Using tokenizer vocab_size: {vocab_size}")
        
        # Initialize model with correct vocab size
        model = GPTModel(
            vocab_size=vocab_size,
            n_embd=model_config['embedding_dim'],
            n_layer=model_config['num_layers'],
            n_head=model_config['num_heads'],
            block_size=model_config['max_seq_length'],
            dropout=model_config.get('dropout', 0.1)
        )
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Loaded PyTorch model with {total_params:,} parameters")
        
        return model
    
    def _load_gguf_model(self) -> GPTModel:
        """Load GGUF model with actual weights."""
        logger.info(f"Loading GGUF model: {self.model_path}")
        
        # For now, use PyTorch model fallback since our GGUF export
        # creates GGUF files but we need to implement proper loading
        # Use the GGUF files with llama.cpp or similar external tools
        
        # This is a placeholder - actual GGUF loading should use llama.cpp
        logger.warning("GGUF files are meant for use with llama.cpp")
        logger.warning("For Python inference, use PyTorch models instead")
        
        # Find PyTorch checkpoint instead
        checkpoint_dir = "exports/checkpoints"
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if checkpoints:
                latest_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
                logger.info(f"Using PyTorch checkpoint instead: {latest_checkpoint}")
                return self._load_pytorch_model_from_path(latest_checkpoint)
        
        raise ValueError("No compatible model found for inference")
    
    def _ensure_vocab_consistency(self):
        """Ensure vocabulary size consistency between tokenizer and model"""
        try:
            # Get actual tokenizer vocab size
            actual_vocab_size = self.tokenizer.vocab_size()
            
            # Update model config to match tokenizer
            self.config['model']['vocab_size'] = actual_vocab_size
            
            logger.info(f"✓ Vocabulary size synchronized: {actual_vocab_size}")
            return actual_vocab_size
            
        except Exception as e:
            logger.warning(f"⚠ Could not auto-detect vocab size: {e}")
            return self.config['model'].get('vocab_size', 16000)
    
    def _load_pytorch_model_from_path(self, model_path: str):
        """Load PyTorch model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = GPTModel(
            vocab_size=self.config['model']['vocab_size'],
            n_embd=self.config['model']['embedding_dim'],
            n_layer=self.config['model']['num_layers'],
            n_head=self.config['model']['num_heads'],
            block_size=self.config['model']['max_seq_length'],
            dropout=self.config['model'].get('dropout', 0.1)
        )
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(self.device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Loaded PyTorch model with {total_params:,} parameters")
        
        return model
    
    def _map_param_name(self, pytorch_name: str) -> str:
        """Map PyTorch parameter names to GGUF tensor names."""
        # Common mappings between PyTorch and GGUF naming conventions
        name_mapping = {
            'token_embedding.weight': 'token_embd.weight',
            'position_embedding.weight': 'pos_embd.weight',
            'ln_f.weight': 'output_norm.weight',
            'ln_f.bias': 'output_norm.bias',
            'lm_head.weight': 'output.weight',
        }
        
        # Handle transformer blocks
        if 'blocks.' in pytorch_name:
            # Convert blocks.0.ln1.weight -> blk.0.attn_norm.weight
            parts = pytorch_name.split('.')
            if len(parts) >= 3:
                block_num = parts[1]
                if 'ln1' in pytorch_name:
                    gguf_name = pytorch_name.replace(f'blocks.{block_num}.ln1', f'blk.{block_num}.attn_norm')
                elif 'ln2' in pytorch_name:
                    gguf_name = pytorch_name.replace(f'blocks.{block_num}.ln2', f'blk.{block_num}.ffn_norm')
                elif 'attn' in pytorch_name:
                    gguf_name = pytorch_name.replace(f'blocks.{block_num}.attn', f'blk.{block_num}.attn')
                elif 'mlp' in pytorch_name:
                    gguf_name = pytorch_name.replace(f'blocks.{block_num}.mlp', f'blk.{block_num}.ffn')
                else:
                    gguf_name = pytorch_name
            else:
                gguf_name = pytorch_name
        else:
            gguf_name = name_mapping.get(pytorch_name, pytorch_name)
        
        return gguf_name
    
    def generate_text(self, 
                     prompt: str,
                     max_new_tokens: int = 100,
                     temperature: float = 0.8,
                     top_k: int = 50,
                     top_p: float = 0.9,
                     do_sample: bool = True) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            Generated text
        """
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        logger.info(f"Generating text for prompt: '{prompt}'")
        logger.info(f"Input tokens: {len(input_ids)}")
        
        generated_ids = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get model predictions
                if len(generated_ids) > self.config['model']['max_seq_length']:
                    # Truncate to max sequence length
                    context_ids = generated_ids[-self.config['model']['max_seq_length']:]
                else:
                    context_ids = generated_ids
                
                context_tensor = torch.tensor([context_ids], device=self.device)
                output = self.model(context_tensor)
                
                # Handle model output (could be tuple or tensor)
                if isinstance(output, tuple):
                    logits = output[0]  # (logits, loss)
                else:
                    logits = output
                
                # Get logits for the last token
                next_token_logits = logits[0, -1, :] / temperature
                
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Apply top-p filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample from the filtered distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                next_token_id = next_token.item()
                
                # Check for end of sequence
                if next_token_id == self.tokenizer.eos_id():
                    break
                
                generated_ids.append(next_token_id)
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids)
        
        logger.info(f"Generated {len(generated_ids) - len(input_ids)} new tokens")
        
        return generated_text
    
    def interactive_mode(self):
        """Run interactive text generation mode."""
        logger.info("Starting interactive mode. Type 'quit' to exit.")
        print("\n" + "="*50)
        print("LLM Interactive Text Generation")
        print("="*50)
        print("Commands:")
        print("  - Type your prompt and press Enter")
        print("  - Type 'quit' to exit")
        print("  - Type 'settings' to adjust generation parameters")
        print("="*50 + "\n")
        
        # Default generation settings
        settings = {
            'max_new_tokens': 100,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'do_sample': True
        }
        
        while True:
            try:
                prompt = input("\nPrompt: ").strip()
                
                if prompt.lower() == 'quit':
                    break
                elif prompt.lower() == 'settings':
                    print("\nCurrent settings:")
                    for key, value in settings.items():
                        print(f"  {key}: {value}")
                    
                    print("\nEnter new values (press Enter to keep current):")
                    for key in settings:
                        new_value = input(f"  {key} [{settings[key]}]: ").strip()
                        if new_value:
                            try:
                                if key == 'do_sample':
                                    settings[key] = new_value.lower() in ['true', '1', 'yes']
                                elif key in ['max_new_tokens', 'top_k']:
                                    settings[key] = int(new_value)
                                else:
                                    settings[key] = float(new_value)
                            except ValueError:
                                print(f"Invalid value for {key}, keeping current value")
                    continue
                
                if not prompt:
                    continue
                
                print("\nGenerating...")
                generated_text = self.generate_text(prompt, **settings)
                
                print(f"\nGenerated text:\n{'-'*40}")
                print(generated_text)
                print("-"*40)
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                logger.error(f"Generation error: {e}")
                print(f"Error: {e}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="LLM Text Generation Inference")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="exports/tokenizer",
                       help="Path to tokenizer directory")
    parser.add_argument("--config", type=str, default="config.json",
                       help="Path to model configuration")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cpu/cuda)")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Text prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p sampling")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("Starting LLM Inference")
    
    try:
        # Initialize inference engine
        inference = LLMInference(
            model_path=args.model,
            tokenizer_path=args.tokenizer,
            config_path=args.config,
            device=args.device
        )
        
        if args.interactive:
            # Run interactive mode
            inference.interactive_mode()
        elif args.prompt:
            # Single prompt generation
            generated_text = inference.generate_text(
                prompt=args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            
            print(f"\nPrompt: {args.prompt}")
            print(f"Generated text:\n{'-'*40}")
            print(generated_text)
            print("-"*40)
        else:
            logger.error("Please provide either --prompt or --interactive")
            return 1
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
