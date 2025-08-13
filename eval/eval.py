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
import torch
import json
from typing import Dict, Any, List, Optional
from loguru import logger

# Add project root to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from model.gpt_model import GPTModel
from training.dataset import TextDataset, MultiFileDataset, create_dataloader
from training.utils import synchronize_vocab_size
from training.utils import (
    DeviceManager, setup_logging, 
    count_parameters, format_time
)

# Tokenizer import
try:
    import sentencepiece as spm
except ImportError:
    logger.error("SentencePiece not available. Please install with: pip install sentencepiece")
    sys.exit(1)


class ModelEvaluator:
    """Comprehensive model evaluation class with hallucination detection."""
    
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
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
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
        """Load model from checkpoint with dynamic vocabulary handling."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Extract model configuration from checkpoint if available
            if 'config' in checkpoint and checkpoint['config']:
                model_config = checkpoint['config']['model']
            elif self.config:
                if 'model' in self.config:
                    model_config = self.config['model']
                else:
                    # Handle case where config is flat structure
                    model_config = self.config
            else:
                logger.error("No model configuration found")
                raise ValueError("No model configuration found")
            
            # Use centralized vocabulary consistency utility
            vocab_size = synchronize_vocab_size(
                {'model': model_config}, 
                self.model_path
            )
            
            # Override model config with synchronized vocab size
            model_config = dict(model_config)
            model_config['vocab_size'] = vocab_size
                
            model = GPTModel(
                vocab_size=vocab_size,
                n_layer=model_config.get('n_layer', model_config.get('num_layers', 6)),
                n_head=model_config.get('n_head', model_config.get('num_heads', 6)),
                n_embd=model_config.get('n_embd', model_config.get('embedding_dim', 512)),
                block_size=model_config.get('block_size', model_config.get('max_seq_length', 256)),
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
    
    def generate_text(self, prompt: str,
                     max_new_tokens: int = 100,
                     temperature: float = 1.0,
                     top_k: Optional[int] = 50,
                     top_p: Optional[float] = 0.9,
                     num_samples: int = 1,
                     validate_responses: bool = True):
        """
        Generate text samples from prompt with quality validation.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            num_samples: Number of samples to generate
            validate_responses: Whether to validate generated responses
            
        Returns:
            List of generated text samples with quality scores
        """
        if self.model is None:
            self.load_model()
        if self.tokenizer is None:
            self.load_tokenizer()
            
        results = []
        
        for _ in range(num_samples):
            try:
                # Encode prompt
                prompt_tokens = self.tokenizer.encode(prompt, out_type=int)
                prompt_tensor = torch.tensor([prompt_tokens], device=self.device)
                
                # Generate
                with torch.no_grad():
                    generated = self.model.generate(
                        prompt_tensor,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p
                    )
                
                # Decode
                generated_tokens = generated[0][len(prompt_tokens):].tolist()
                generated_text = self.tokenizer.decode(generated_tokens)
                
                # Clean up text
                generated_text = generated_text.strip()
                
                # Validate response quality
                validation_result = {
                    'text': generated_text,
                    'prompt': prompt,
                    'quality_score': 1.0,
                    'hallucination_detected': False,
                    'coherence_score': 1.0,
                    'repetition_ratio': 0.0,
                    'factuality_score': 1.0,
                    'issues': []
                }
                
                if validate_responses:
                    validation_result = self.validate_response_quality(
                        prompt, generated_text
                    )
                
                results.append(validation_result)
                
            except Exception as e:
                logger.error(f"Text generation failed: {e}")
                results.append({
                    'text': f"[ERROR: {str(e)}]",
                    'prompt': prompt,
                    'quality_score': 0.0,
                    'hallucination_detected': True,
                    'issues': [str(e)]
                })
        
        return results
    
    def validate_response_quality(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Validate response quality and detect hallucinations.
        
        Args:
            prompt: Original prompt
            response: Generated response
            
        Returns:
            Dictionary with quality metrics and issues
        """
        issues = []
        quality_score = 1.0
        hallucination_detected = False
        
        # Basic quality checks
        if not response or len(response.strip()) < 10:
            issues.append("Response too short or empty")
            quality_score -= 0.3
        
        # Repetition detection
        repetition_ratio = self.calculate_repetition_ratio(response)
        if repetition_ratio > self.quality_thresholds['max_repetition_ratio']:
            issues.append(f"High repetition: {repetition_ratio:.2f}")
            quality_score -= 0.2
        
        # Coherence check
        coherence_score = self.calculate_coherence_score(prompt, response)
        if coherence_score < self.quality_thresholds['min_coherence_score']:
            issues.append(f"Low coherence: {coherence_score:.2f}")
            quality_score -= 0.2
        
        # Factuality check (basic)
        factuality_score = self.calculate_factuality_score(prompt, response)
        if factuality_score < self.quality_thresholds['min_factuality_score']:
            issues.append(f"Low factuality: {factuality_score:.2f}")
            quality_score -= 0.3
            hallucination_detected = True
        
        # Hallucination patterns
        hallucination_patterns = self.detect_hallucination_patterns(response)
        if hallucination_patterns:
            issues.extend(hallucination_patterns)
            hallucination_detected = True
            quality_score -= 0.4
        
        return {
            'text': response,
            'prompt': prompt,
            'quality_score': max(0.0, quality_score),
            'hallucination_detected': hallucination_detected,
            'coherence_score': coherence_score,
            'repetition_ratio': repetition_ratio,
            'factuality_score': factuality_score,
            'issues': issues
        }

    def calculate_repetition_ratio(self, text: str) -> float:
        """Calculate repetition ratio in text."""
        words = text.lower().split()
        if len(words) < 3:
            return 0.0
        
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        max_repeats = max(word_counts.values()) if word_counts else 1
        repetition_ratio = (max_repeats - 1) / len(words)
        return repetition_ratio

    def calculate_coherence_score(self, prompt: str, response: str) -> float:
        """Calculate coherence between prompt and response."""
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        if not prompt_words or not response_words:
            return 0.0
        
        # Check for keyword overlap
        overlap = len(prompt_words.intersection(response_words))
        coherence_score = overlap / max(len(prompt_words), len(response_words))
        
        # Boost score if response addresses prompt topic
        prompt_topics = ['artificial', 'intelligence', 'ai', 'future', 'technology']
        response_lower = response.lower()
        
        topic_relevance = sum(1 for topic in prompt_topics if topic in response_lower)
        coherence_score += (topic_relevance * 0.2)
        
        return min(1.0, coherence_score)

    def calculate_factuality_score(self, prompt: str, response: str) -> float:
        """Calculate basic factuality score."""
        # Simple heuristics for factuality
        factual_indicators = [
            'according to', 'research shows', 'studies indicate',
            'data suggests', 'evidence indicates', 'reported by'
        ]
        
        speculative_indicators = [
            'might be', 'could be', 'possibly', 'perhaps',
            'maybe', 'i think', 'i believe', 'in my opinion'
        ]
        
        factual_score = 0.5  # Base score
        
        # Check for factual indicators
        for indicator in factual_indicators:
            if indicator in response.lower():
                factual_score += 0.2
        
        # Penalize speculative language
        for indicator in speculative_indicators:
            if indicator in response.lower():
                factual_score -= 0.1
        
        # Check for specific claims (numbers, dates, names)
        import re
        numbers = re.findall(r'\d+', response)
        dates = re.findall(r'\d{4}', response)  # Years
        
        if numbers or dates:
            factual_score += 0.2
        
        return max(0.0, min(1.0, factual_score))

    def detect_hallucination_patterns(self, response: str) -> List[str]:
        """Detect common hallucination patterns."""
        issues = []
        
        # Common hallucination patterns
        hallucination_patterns = [
            'i am an ai assistant',
            'as an ai language model',
            'i am programmed to',
            'my training data',
            'i was trained on',
            'as a large language model'
        ]
        
        response_lower = response.lower()
        
        for pattern in hallucination_patterns:
            if pattern in response_lower:
                issues.append(f"Self-referential hallucination: {pattern}")
        
        # Check for made-up facts
        if 'according to' in response_lower and len(response) < 50:
            issues.append("Potentially fabricated attribution")
        
        # Check for contradictory statements
        if 'however' in response_lower and 'but' in response_lower:
            issues.append("Potential contradiction detected")
        
        return issues

    def benchmark_inference(self, sequence_length: int = 256,
                           batch_size: int = 1,
                           num_iterations: int = 100): 
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
    
    def comprehensive_evaluation(self, test_data_path: Optional[str] = None,
                               prompts: Optional[List[str]] = None,
                               quality_threshold: float = 0.7):
        """
        Run comprehensive evaluation with quality assessment.
        
        Args:
            test_data_path: Path to test data
            prompts: List of prompts for generation
            quality_threshold: Minimum quality score for acceptable responses
            
        Returns:
            Comprehensive evaluation results with quality metrics
        """
        if self.model is None:
            self.load_model()
        if self.tokenizer is None:
            self.load_tokenizer()
            
        results = {
            'model_info': {},
            'perplexity': None,
            'generation': [],
            'benchmark': None,
            'quality_summary': {},
            'hallucination_summary': {},
            'recommendations': []
        }
        
        logger.info("Starting comprehensive evaluation...")
        
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
            config_path="config.json"
        )
        
        # Test prompts
        test_prompts = [
            "The future of artificial intelligence",
            "In a world where",
            "The most important thing to remember",
            "Once upon a time",
            "Explain quantum computing",
            "What are the benefits of renewable energy",
            "Describe the process of photosynthesis"
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

