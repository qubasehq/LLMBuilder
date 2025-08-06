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
from typing import Optional, List

import torch
import torch.nn.functional as F
from loguru import logger
import sentencepiece as spm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from model.gpt_model import GPTModel
from training.utils import setup_logging


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
            model_path: Path to trained model checkpoint
            tokenizer_path: Path to tokenizer directory
            config_path: Path to model configuration
            device: Device to run inference on (auto-detect if None)
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.config_path = config_path
        
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
        
        # Load configuration
        self.config = self._load_config()
        
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
        tokenizer_model_path = os.path.join(self.tokenizer_path, "tokenizer.model")
        
        if not os.path.exists(tokenizer_model_path):
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
        """Load the trained model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        try:
            # Initialize model
            model_config = self.config['model']
            model = GPTModel(
                vocab_size=model_config['vocab_size'],
                embedding_dim=model_config['embedding_dim'],
                num_layers=model_config['num_layers'],
                num_heads=model_config['num_heads'],
                hidden_dim=model_config['hidden_dim'],
                max_seq_length=model_config['max_seq_length'],
                dropout=model_config.get('dropout', 0.1),
                use_bias=model_config.get('use_bias', True),
                tie_weights=model_config.get('tie_weights', True)
            )
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Loaded model with {total_params:,} parameters")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
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
                logits = self.model(context_tensor)
                
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
