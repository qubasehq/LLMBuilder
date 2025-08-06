"""
Model evaluation script for LLM.
Provides comprehensive evaluation including perplexity, generation quality, and performance metrics.
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from model.gpt_model import GPTModel
from training.dataset import TextDataset, create_dataloader
from training.utils import (
    ConfigManager, DeviceManager, setup_logging, 
    count_parameters, format_time
)

# Tokenizer import
try:
    import sentencepiece as spm
except ImportError:
    logger.error("SentencePiece not available. Please install with: pip install sentencepiece")
    sys.exit(1)


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, 
                 model_path: str,
                 tokenizer_path: str,
                 config_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize model evaluator.
        
        Args:
            model_path: Path to trained model checkpoint
            tokenizer_path: Path to tokenizer model
            config_path: Path to config file (optional)
            device: Device to use (optional, auto-detected if None)
        """
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        
        # Setup logging
        setup_logging(log_dir="logs", level="INFO")
        
        # Load configuration
        if config_path:
            self.config = ConfigManager.load_config(config_path)
        else:
            self.config = None
        
        # Setup device
        if device:
            self.device = torch.device(device)
        else:
            self.device = DeviceManager.get_device(prefer_cpu=True)
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        
        logger.info("Model evaluator initialized")
    
    def load_model(self) -> GPTModel:
        """Load trained model from checkpoint."""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Get model configuration
            if 'config' in checkpoint:
                model_config = checkpoint['config']['model']
            elif self.config:
                model_config = self.config['model']
            else:
                raise ValueError("No model configuration found")
            
            # Create model
            model = GPTModel(
                vocab_size=model_config['vocab_size'],
                n_layer=model_config['n_layer'],
                n_head=model_config['n_head'],
                n_embd=model_config['n_embd'],
                block_size=model_config['block_size'],
                dropout=0.0  # No dropout during evaluation
            )
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            # Move to device and set to eval mode
            model = model.to(self.device)
            model.eval()
            
            # Log model info
            n_params = count_parameters(model)
            logger.info(f"Model loaded successfully: {n_params:,} parameters")
            
            if 'epoch' in checkpoint:
                logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
            if 'metrics' in checkpoint:
                logger.info(f"Checkpoint metrics: {checkpoint['metrics']}")
            
            self.model = model
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_tokenizer(self) -> spm.SentencePieceProcessor:
        """Load tokenizer."""
        logger.info(f"Loading tokenizer from {self.tokenizer_path}")
        
        try:
            tokenizer = spm.SentencePieceProcessor()
            tokenizer.load(str(self.tokenizer_path))
            
            logger.info(f"Tokenizer loaded: vocab size {tokenizer.get_piece_size()}")
            
            self.tokenizer = tokenizer
            return tokenizer
            
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def evaluate_perplexity(self, 
                           data_path: str,
                           batch_size: int = 16,
                           max_samples: Optional[int] = None) -> Dict[str, float]:
        """Evaluate model perplexity on dataset."""
        logger.info("Evaluating perplexity...")
        
        if not self.model:
            raise ValueError("Model not loaded")
        
        try:
            # Load dataset
            dataset = TextDataset(
                data_path=data_path,
                block_size=self.model.block_size
            )
            
            # Limit samples if specified
            if max_samples and len(dataset) > max_samples:
                dataset.num_samples = max_samples
            
            # Create dataloader
            dataloader = create_dataloader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=0
            )
            
            total_loss = 0.0
            total_tokens = 0
            num_batches = 0
            
            start_time = time.time()
            
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    logits, loss = self.model(input_ids, labels)
                    
                    # Accumulate metrics
                    total_loss += loss.item()
                    total_tokens += labels.numel()
                    num_batches += 1
                    
                    if num_batches % 100 == 0:
                        logger.info(f"Processed {num_batches} batches")
            
            # Calculate metrics
            avg_loss = total_loss / num_batches
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            eval_time = time.time() - start_time
            tokens_per_sec = total_tokens / eval_time
            
            results = {
                'loss': avg_loss,
                'perplexity': perplexity,
                'total_tokens': total_tokens,
                'eval_time': eval_time,
                'tokens_per_sec': tokens_per_sec
            }
            
            logger.info(f"Perplexity evaluation complete:")
            logger.info(f"  Loss: {avg_loss:.4f}")
            logger.info(f"  Perplexity: {perplexity:.2f}")
            logger.info(f"  Tokens/sec: {tokens_per_sec:.0f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating perplexity: {e}")
            raise
    
    def generate_text(self, 
                     prompt: str,
                     max_new_tokens: int = 100,
                     temperature: float = 1.0,
                     top_k: Optional[int] = 50,
                     num_samples: int = 1) -> List[str]:
        """Generate text samples from prompt."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be loaded")
        
        logger.info(f"Generating text with prompt: '{prompt}'")
        
        try:
            # Encode prompt
            prompt_tokens = self.tokenizer.encode(prompt, out_type=int)
            input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
            
            generated_texts = []
            
            for i in range(num_samples):
                # Generate
                with torch.no_grad():
                    generated = self.model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k
                    )
                
                # Decode
                generated_tokens = generated[0].cpu().tolist()
                generated_text = self.tokenizer.decode(generated_tokens)
                generated_texts.append(generated_text)
                
                logger.info(f"Sample {i+1}: {generated_text}")
            
            return generated_texts
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def benchmark_inference(self, 
                           sequence_length: int = 256,
                           batch_size: int = 1,
                           num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark model inference speed."""
        if not self.model:
            raise ValueError("Model not loaded")
        
        logger.info(f"Benchmarking inference speed...")
        
        try:
            # Create dummy input
            dummy_input = torch.randint(
                0, self.model.vocab_size, 
                (batch_size, sequence_length),
                device=self.device
            )
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = self.model(dummy_input)
            
            # Benchmark
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = self.model(dummy_input)
            
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_time_per_batch = total_time / num_iterations
            tokens_per_sec = (batch_size * sequence_length * num_iterations) / total_time
            
            results = {
                'total_time': total_time,
                'avg_time_per_batch': avg_time_per_batch,
                'tokens_per_sec': tokens_per_sec,
                'iterations': num_iterations,
                'batch_size': batch_size,
                'sequence_length': sequence_length
            }
            
            logger.info(f"Inference benchmark complete:")
            logger.info(f"  Avg time per batch: {avg_time_per_batch*1000:.2f}ms")
            logger.info(f"  Tokens/sec: {tokens_per_sec:.0f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error benchmarking inference: {e}")
            raise
    
    def comprehensive_evaluation(self, 
                               test_data_path: Optional[str] = None,
                               prompts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive evaluation."""
        logger.info("Starting comprehensive evaluation...")
        
        # Load model and tokenizer
        self.load_model()
        self.load_tokenizer()
        
        results = {}
        
        # Perplexity evaluation
        if test_data_path and Path(test_data_path).exists():
            try:
                perplexity_results = self.evaluate_perplexity(test_data_path)
                results['perplexity'] = perplexity_results
            except Exception as e:
                logger.error(f"Perplexity evaluation failed: {e}")
                results['perplexity_error'] = str(e)
        
        # Text generation
        if prompts:
            try:
                generation_results = []
                for prompt in prompts:
                    generated = self.generate_text(
                        prompt=prompt,
                        max_new_tokens=50,
                        temperature=0.8,
                        top_k=50,
                        num_samples=1
                    )
                    generation_results.append({
                        'prompt': prompt,
                        'generated': generated[0]
                    })
                results['generation'] = generation_results
            except Exception as e:
                logger.error(f"Text generation failed: {e}")
                results['generation_error'] = str(e)
        
        # Inference benchmark
        try:
            benchmark_results = self.benchmark_inference()
            results['benchmark'] = benchmark_results
        except Exception as e:
            logger.error(f"Inference benchmark failed: {e}")
            results['benchmark_error'] = str(e)
        
        # Model info
        results['model_info'] = {
            'parameters': count_parameters(self.model),
            'vocab_size': self.model.vocab_size,
            'n_layer': self.model.n_layer,
            'n_head': self.model.n_head,
            'n_embd': self.model.n_embd,
            'block_size': self.model.block_size,
            'device': str(self.device)
        }
        
        logger.info("Comprehensive evaluation complete")
        return results


def main():
    """Main entry point."""
    try:
        # Default paths
        model_path = "exports/checkpoints/best_model.pt"
        tokenizer_path = "tokenizer/tokenizer.model"
        test_data_path = "data/tokens/tokens.pt"
        
        # Check if files exist
        if not Path(model_path).exists():
            logger.error(f"Model not found: {model_path}")
            logger.info("Please train a model first using: python training/train.py")
            sys.exit(1)
        
        if not Path(tokenizer_path).exists():
            logger.error(f"Tokenizer not found: {tokenizer_path}")
            logger.info("Please train tokenizer first using: python training/train_tokenizer.py")
            sys.exit(1)
        
        # Initialize evaluator
        evaluator = ModelEvaluator(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            config_path="training/config.yaml"
        )
        
        # Test prompts
        test_prompts = [
            "The future of artificial intelligence",
            "In a world where",
            "The most important thing to remember",
            "Once upon a time"
        ]
        
        # Run comprehensive evaluation
        results = evaluator.comprehensive_evaluation(
            test_data_path=test_data_path if Path(test_data_path).exists() else None,
            prompts=test_prompts
        )
        
        # Print summary
        logger.info("=== Evaluation Summary ===")
        
        if 'model_info' in results:
            info = results['model_info']
            logger.info(f"Model: {info['parameters']:,} parameters")
            logger.info(f"Architecture: {info['n_layer']}L-{info['n_head']}H-{info['n_embd']}D")
            logger.info(f"Device: {info['device']}")
        
        if 'perplexity' in results:
            ppl = results['perplexity']
            logger.info(f"Perplexity: {ppl['perplexity']:.2f}")
            logger.info(f"Loss: {ppl['loss']:.4f}")
        
        if 'benchmark' in results:
            bench = results['benchmark']
            logger.info(f"Inference speed: {bench['tokens_per_sec']:.0f} tokens/sec")
        
        if 'generation' in results:
            logger.info(f"Generated {len(results['generation'])} text samples")
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

