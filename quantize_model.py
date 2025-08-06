#!/usr/bin/env python3
"""
Model quantization CLI script.
Provides easy-to-use interface for quantizing trained models.
"""

import argparse
import torch
from pathlib import Path
from training.quantization import QuantizationManager, QuantConfig
from loguru import logger


def quantize_model(
    model_path: str,
    output_path: str,
    bits: int = 8,
    scheme: str = "symmetric",
    granularity: str = "per_tensor"
):
    """Quantize a trained model."""
    
    try:
        logger.info(f"Starting quantization: {model_path} -> {output_path}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        model = checkpoint['model']
        config = checkpoint['config']
        
        # Create quantization config
        quant_config = QuantConfig(
            bits=bits,
            scheme=scheme,
            granularity=granularity
        )
        
        # Quantize model
        quantized_model = QuantizationManager.quantize_model(model, quant_config)
        
        # Calculate size reduction
        original_size = QuantizationManager.get_model_size(model)
        quantized_size = QuantizationManager.get_model_size(quantized_model)
        
        # Save quantized model
        torch.save({
            'model': quantized_model,
            'config': config,
            'quant_config': quant_config,
            'original_size_mb': original_size['model_size_mb'],
            'quantized_size_mb': quantized_size['model_size_mb'],
            'compression_ratio': original_size['model_size_mb'] / quantized_size['model_size_mb']
        }, output_path)
        
        logger.success(f"Model quantized successfully!")
        logger.info(f"Original size: {original_size['model_size_mb']:.2f} MB")
        logger.info(f"Quantized size: {quantized_size['model_size_mb']:.2f} MB")
        logger.info(f"Compression ratio: {original_size['model_size_mb']/quantized_size['model_size_mb']:.2f}x")
        
        return True
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        return False


def main():
    """Main CLI function."""
    
    parser = argparse.ArgumentParser(description="Quantize LLM models")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", required=True, help="Output path for quantized model")
    parser.add_argument("--bits", type=int, choices=[4, 8, 16], default=8,
                       help="Quantization bits (4, 8, or 16)")
    parser.add_argument("--scheme", choices=["symmetric", "asymmetric", "dynamic"], 
                       default="symmetric", help="Quantization scheme")
    parser.add_argument("--granularity", choices=["per_tensor", "per_channel", "per_group"],
                       default="per_tensor", help="Quantization granularity")
    parser.add_argument("--list-sizes", action="store_true", 
                       help="List model sizes without quantizing")
    
    args = parser.parse_args()
    
    if args.list_sizes:
        # Just list model information
        try:
            checkpoint = torch.load(args.model, map_location='cpu')
            model = checkpoint['model']
            size_info = QuantizationManager.get_model_size(model)
            
            logger.info(f"Model: {args.model}")
            logger.info(f"Total parameters: {size_info['total_params']:,}")
            logger.info(f"Model size: {size_info['model_size_mb']:.2f} MB")
            
            # Estimate quantized sizes
            for bits in [4, 8, 16]:
                estimated_size = QuantizationManager.estimate_quantized_size(
                    size_info['model_size_mb'], bits
                )
                logger.info(f"Estimated {bits}-bit size: {estimated_size:.2f} MB")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return 1
    else:
        # Perform quantization
        success = quantize_model(
            args.model,
            args.output,
            args.bits,
            args.scheme,
            args.granularity
        )
        
        return 0 if success else 1


if __name__ == "__main__":
    exit(main())
