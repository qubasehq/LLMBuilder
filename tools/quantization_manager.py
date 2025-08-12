"""
Comprehensive quantization manager for GGUF models.
Supports multiple quantization formats with quality validation and optimization.
"""

import os
import struct
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import time
import json
from loguru import logger


class QuantizationType(Enum):
    """Supported quantization types."""
    F32 = "f32"
    F16 = "f16"
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"
    Q4_1 = "q4_1"
    Q5_0 = "q5_0"
    Q5_1 = "q5_1"


@dataclass
class QuantizationConfig:
    """Configuration for quantization process."""
    quantization_type: QuantizationType
    block_size: int = 32
    quality_threshold: float = 0.95  # Minimum quality score
    skip_layers: List[str] = None  # Layers to skip quantization
    force_layers: List[str] = None  # Layers to force quantization
    
    def __post_init__(self):
        if self.skip_layers is None:
            self.skip_layers = []
        if self.force_layers is None:
            self.force_layers = []


@dataclass
class QuantizationResult:
    """Result of quantization process."""
    original_size: int
    quantized_size: int
    compression_ratio: float
    quality_score: float
    processing_time: float
    tensor_results: Dict[str, Dict[str, Any]]
    
    @property
    def size_reduction_mb(self) -> float:
        """Size reduction in MB."""
        return (self.original_size - self.quantized_size) / (1024 * 1024)
    
    @property
    def size_reduction_percent(self) -> float:
        """Size reduction as percentage."""
        return (1 - self.quantized_size / self.original_size) * 100


class QuantizationManager:
    """Comprehensive quantization manager with quality validation."""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.quantized_tensors: Dict[str, np.ndarray] = {}
        self.tensor_stats: Dict[str, Dict[str, Any]] = {}
        self.progress_callback: Optional[Callable[[str, float], None]] = None   
 
    def set_progress_callback(self, callback: Callable[[str, float], None]) -> None:
        """Set progress callback for monitoring quantization progress."""
        self.progress_callback = callback
    
    def _report_progress(self, message: str, progress: float) -> None:
        """Report progress to callback if available."""
        if self.progress_callback:
            self.progress_callback(message, progress)
        else:
            logger.info(f"{message} ({progress:.1f}%)")
    
    def quantize_tensor(self, tensor: torch.Tensor, name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize a single tensor and return quantized data with statistics."""
        start_time = time.time()
        original_tensor = tensor.detach().cpu().float()
        
        # Choose quantization method
        if self.config.quantization_type == QuantizationType.F16:
            quantized_data, stats = self._quantize_f16(original_tensor)
        elif self.config.quantization_type == QuantizationType.Q8_0:
            quantized_data, stats = self._quantize_q8_0(original_tensor)
        elif self.config.quantization_type == QuantizationType.Q4_0:
            quantized_data, stats = self._quantize_q4_0(original_tensor)
        elif self.config.quantization_type == QuantizationType.Q4_1:
            quantized_data, stats = self._quantize_q4_1(original_tensor)
        elif self.config.quantization_type == QuantizationType.Q5_0:
            quantized_data, stats = self._quantize_q5_0(original_tensor)
        elif self.config.quantization_type == QuantizationType.Q5_1:
            quantized_data, stats = self._quantize_q5_1(original_tensor)
        else:  # F32
            quantized_data, stats = self._quantize_f32(original_tensor)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Add common statistics
        stats.update({
            'name': name,
            'original_shape': tuple(original_tensor.shape),
            'original_size': original_tensor.numel() * 4,  # F32 size
            'quantized_size': len(quantized_data),
            'compression_ratio': (original_tensor.numel() * 4) / len(quantized_data),
            'processing_time': processing_time,
            'quantization_type': self.config.quantization_type.value
        })
        
        return quantized_data, stats
    
    def _quantize_f32(self, tensor: torch.Tensor) -> Tuple[np.ndarray, Dict[str, Any]]:
        """No quantization - keep as F32."""
        data = tensor.numpy().astype(np.float32).tobytes()
        stats = {
            'quality_score': 1.0,
            'error_metrics': {'mse': 0.0, 'max_error': 0.0}
        }
        return np.frombuffer(data, dtype=np.uint8), stats
    
    def _quantize_f16(self, tensor: torch.Tensor) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize to F16 format."""
        f16_tensor = tensor.half()
        reconstructed = f16_tensor.float()
        
        # Calculate quality metrics
        mse = torch.mean((tensor - reconstructed) ** 2).item()
        max_error = torch.max(torch.abs(tensor - reconstructed)).item()
        quality_score = self._calculate_quality_score(tensor, reconstructed)
        
        data = f16_tensor.numpy().astype(np.float16).tobytes()
        stats = {
            'quality_score': quality_score,
            'error_metrics': {'mse': mse, 'max_error': max_error}
        }
        
        return np.frombuffer(data, dtype=np.uint8), stats 
   
    def _quantize_q8_0(self, tensor: torch.Tensor) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize to Q8_0 format (8-bit with block-wise scaling)."""
        tensor_flat = tensor.flatten()
        n = tensor_flat.numel()
        block_size = self.config.block_size
        n_blocks = (n + block_size - 1) // block_size
        
        # Each block: 32 int8 values + 1 float16 scale = 34 bytes
        quantized = np.zeros(n_blocks * 34, dtype=np.uint8)
        reconstructed_values = []
        
        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, n)
            block = tensor_flat[start_idx:end_idx]
            
            # Pad block if necessary
            if len(block) < block_size:
                block = torch.cat([block, torch.zeros(block_size - len(block))])
            
            # Calculate scale
            absmax = torch.max(torch.abs(block))
            if absmax > 0:
                scale = absmax / 127.0
                quantized_block = torch.round(block / scale).clamp(-128, 127).to(torch.int8)
            else:
                scale = torch.tensor(1.0)
                quantized_block = torch.zeros(block_size, dtype=torch.int8)
            
            # Store quantized values and scale
            block_start = i * 34
            quantized[block_start:block_start+32] = quantized_block.numpy().astype(np.uint8)
            quantized[block_start+32:block_start+34] = np.frombuffer(
                struct.pack('<e', scale.item()), dtype=np.uint8
            )
            
            # Reconstruct for quality calculation
            reconstructed_block = quantized_block.float() * scale
            reconstructed_values.extend(reconstructed_block[:len(block)-len(torch.zeros(block_size-len(block)))])
        
        # Calculate quality metrics
        reconstructed = torch.tensor(reconstructed_values[:n])
        mse = torch.mean((tensor_flat - reconstructed) ** 2).item()
        max_error = torch.max(torch.abs(tensor_flat - reconstructed)).item()
        quality_score = self._calculate_quality_score(tensor_flat, reconstructed)
        
        stats = {
            'quality_score': quality_score,
            'error_metrics': {'mse': mse, 'max_error': max_error},
            'blocks': n_blocks,
            'block_size': block_size
        }
        
        return quantized, stats
    
    def _quantize_q4_0(self, tensor: torch.Tensor) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize to Q4_0 format (4-bit with block-wise scaling)."""
        tensor_flat = tensor.flatten()
        n = tensor_flat.numel()
        block_size = self.config.block_size
        n_blocks = (n + block_size - 1) // block_size
        
        # Each block: 16 nibbles (32 4-bit values) + 1 float16 scale = 18 bytes
        quantized = np.zeros(n_blocks * 18, dtype=np.uint8)
        reconstructed_values = []
        
        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, n)
            block = tensor_flat[start_idx:end_idx]
            
            # Pad block if necessary
            if len(block) < block_size:
                block = torch.cat([block, torch.zeros(block_size - len(block))])
            
            # Calculate scale
            absmax = torch.max(torch.abs(block))
            if absmax > 0:
                scale = absmax / 7.0  # 4-bit signed: -8 to 7
                quantized_block = torch.round(block / scale).clamp(-8, 7).to(torch.int8)
            else:
                scale = torch.tensor(1.0)
                quantized_block = torch.zeros(block_size, dtype=torch.int8)
            
            # Pack 4-bit values into bytes (2 values per byte)
            block_start = i * 18
            for j in range(0, block_size, 2):
                val1 = int(quantized_block[j].item()) & 0xF
                val2 = int(quantized_block[j+1].item()) & 0xF if j+1 < block_size else 0
                quantized[block_start + j//2] = (val2 << 4) | val1
            
            # Store scale
            quantized[block_start+16:block_start+18] = np.frombuffer(
                struct.pack('<e', scale.item()), dtype=np.uint8
            )
            
            # Reconstruct for quality calculation
            reconstructed_block = quantized_block.float() * scale
            actual_block_size = min(block_size, n - start_idx)
            reconstructed_values.extend(reconstructed_block[:actual_block_size])
        
        # Calculate quality metrics
        reconstructed = torch.tensor(reconstructed_values[:n])
        mse = torch.mean((tensor_flat - reconstructed) ** 2).item()
        max_error = torch.max(torch.abs(tensor_flat - reconstructed)).item()
        quality_score = self._calculate_quality_score(tensor_flat, reconstructed)
        
        stats = {
            'quality_score': quality_score,
            'error_metrics': {'mse': mse, 'max_error': max_error},
            'blocks': n_blocks,
            'block_size': block_size
        }
        
        return quantized, stats  
  
    def _quantize_q4_1(self, tensor: torch.Tensor) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize to Q4_1 format (4-bit with scale and min)."""
        tensor_flat = tensor.flatten()
        n = tensor_flat.numel()
        block_size = self.config.block_size
        n_blocks = (n + block_size - 1) // block_size
        
        # Each block: 16 nibbles + 1 float16 scale + 1 float16 min = 20 bytes
        quantized = np.zeros(n_blocks * 20, dtype=np.uint8)
        reconstructed_values = []
        
        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, n)
            block = tensor_flat[start_idx:end_idx]
            
            if len(block) < block_size:
                block = torch.cat([block, torch.zeros(block_size - len(block))])
            
            # Calculate scale and min
            block_min = torch.min(block)
            block_max = torch.max(block)
            scale = (block_max - block_min) / 15.0  # 4-bit unsigned: 0 to 15
            
            if scale > 0:
                quantized_block = torch.round((block - block_min) / scale).clamp(0, 15).to(torch.uint8)
            else:
                scale = torch.tensor(1.0)
                quantized_block = torch.zeros(block_size, dtype=torch.uint8)
            
            # Pack 4-bit values
            block_start = i * 20
            for j in range(0, block_size, 2):
                val1 = int(quantized_block[j].item()) & 0xF
                val2 = int(quantized_block[j+1].item()) & 0xF if j+1 < block_size else 0
                quantized[block_start + j//2] = (val2 << 4) | val1
            
            # Store scale and min
            quantized[block_start+16:block_start+18] = np.frombuffer(
                struct.pack('<e', scale.item()), dtype=np.uint8
            )
            quantized[block_start+18:block_start+20] = np.frombuffer(
                struct.pack('<e', block_min.item()), dtype=np.uint8
            )
            
            # Reconstruct
            reconstructed_block = quantized_block.float() * scale + block_min
            actual_block_size = min(block_size, n - start_idx)
            reconstructed_values.extend(reconstructed_block[:actual_block_size])
        
        reconstructed = torch.tensor(reconstructed_values[:n])
        mse = torch.mean((tensor_flat - reconstructed) ** 2).item()
        max_error = torch.max(torch.abs(tensor_flat - reconstructed)).item()
        quality_score = self._calculate_quality_score(tensor_flat, reconstructed)
        
        stats = {
            'quality_score': quality_score,
            'error_metrics': {'mse': mse, 'max_error': max_error},
            'blocks': n_blocks,
            'block_size': block_size
        }
        
        return quantized, stats
    
    def _quantize_q5_0(self, tensor: torch.Tensor) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize to Q5_0 format (5-bit with block-wise scaling)."""
        tensor_flat = tensor.flatten()
        n = tensor_flat.numel()
        block_size = self.config.block_size
        n_blocks = (n + block_size - 1) // block_size
        
        # Each block: 20 bytes for 5-bit values + 1 float16 scale + 4 bytes for high bits = 26 bytes
        quantized = np.zeros(n_blocks * 26, dtype=np.uint8)
        reconstructed_values = []
        
        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, n)
            block = tensor_flat[start_idx:end_idx]
            
            if len(block) < block_size:
                block = torch.cat([block, torch.zeros(block_size - len(block))])
            
            # Calculate scale
            absmax = torch.max(torch.abs(block))
            if absmax > 0:
                scale = absmax / 15.0  # 5-bit signed: -16 to 15
                quantized_block = torch.round(block / scale).clamp(-16, 15).to(torch.int8)
            else:
                scale = torch.tensor(1.0)
                quantized_block = torch.zeros(block_size, dtype=torch.int8)
            
            # Store 4 lower bits and 1 high bit separately
            block_start = i * 26
            high_bits = 0
            
            for j in range(block_size):
                val = int(quantized_block[j].item())
                low_4_bits = val & 0xF
                high_bit = (val >> 4) & 1
                
                if j % 2 == 0:
                    quantized[block_start + j//2] = low_4_bits
                else:
                    quantized[block_start + j//2] |= (low_4_bits << 4)
                
                if high_bit:
                    high_bits |= (1 << j)
            
            # Store high bits (4 bytes for 32 bits)
            quantized[block_start+16:block_start+20] = np.frombuffer(
                struct.pack('<I', high_bits), dtype=np.uint8
            )
            
            # Store scale
            quantized[block_start+20:block_start+22] = np.frombuffer(
                struct.pack('<e', scale.item()), dtype=np.uint8
            )
            
            # Reconstruct
            reconstructed_block = quantized_block.float() * scale
            actual_block_size = min(block_size, n - start_idx)
            reconstructed_values.extend(reconstructed_block[:actual_block_size])
        
        reconstructed = torch.tensor(reconstructed_values[:n])
        mse = torch.mean((tensor_flat - reconstructed) ** 2).item()
        max_error = torch.max(torch.abs(tensor_flat - reconstructed)).item()
        quality_score = self._calculate_quality_score(tensor_flat, reconstructed)
        
        stats = {
            'quality_score': quality_score,
            'error_metrics': {'mse': mse, 'max_error': max_error},
            'blocks': n_blocks,
            'block_size': block_size
        }
        
        return quantized, stats    

    def _quantize_q5_1(self, tensor: torch.Tensor) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize to Q5_1 format (5-bit with scale and min)."""
        tensor_flat = tensor.flatten()
        n = tensor_flat.numel()
        block_size = self.config.block_size
        n_blocks = (n + block_size - 1) // block_size
        
        # Each block: 20 bytes for 5-bit values + 4 bytes high bits + 2 bytes scale + 2 bytes min = 28 bytes
        quantized = np.zeros(n_blocks * 28, dtype=np.uint8)
        reconstructed_values = []
        
        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, n)
            block = tensor_flat[start_idx:end_idx]
            
            if len(block) < block_size:
                block = torch.cat([block, torch.zeros(block_size - len(block))])
            
            # Calculate scale and min
            block_min = torch.min(block)
            block_max = torch.max(block)
            scale = (block_max - block_min) / 31.0  # 5-bit unsigned: 0 to 31
            
            if scale > 0:
                quantized_block = torch.round((block - block_min) / scale).clamp(0, 31).to(torch.uint8)
            else:
                scale = torch.tensor(1.0)
                quantized_block = torch.zeros(block_size, dtype=torch.uint8)
            
            # Store 4 lower bits and 1 high bit separately
            block_start = i * 28
            high_bits = 0
            
            for j in range(block_size):
                val = int(quantized_block[j].item())
                low_4_bits = val & 0xF
                high_bit = (val >> 4) & 1
                
                if j % 2 == 0:
                    quantized[block_start + j//2] = low_4_bits
                else:
                    quantized[block_start + j//2] |= (low_4_bits << 4)
                
                if high_bit:
                    high_bits |= (1 << j)
            
            # Store high bits, scale, and min
            quantized[block_start+16:block_start+20] = np.frombuffer(
                struct.pack('<I', high_bits), dtype=np.uint8
            )
            quantized[block_start+20:block_start+22] = np.frombuffer(
                struct.pack('<e', scale.item()), dtype=np.uint8
            )
            quantized[block_start+22:block_start+24] = np.frombuffer(
                struct.pack('<e', block_min.item()), dtype=np.uint8
            )
            
            # Reconstruct
            reconstructed_block = quantized_block.float() * scale + block_min
            actual_block_size = min(block_size, n - start_idx)
            reconstructed_values.extend(reconstructed_block[:actual_block_size])
        
        reconstructed = torch.tensor(reconstructed_values[:n])
        mse = torch.mean((tensor_flat - reconstructed) ** 2).item()
        max_error = torch.max(torch.abs(tensor_flat - reconstructed)).item()
        quality_score = self._calculate_quality_score(tensor_flat, reconstructed)
        
        stats = {
            'quality_score': quality_score,
            'error_metrics': {'mse': mse, 'max_error': max_error},
            'blocks': n_blocks,
            'block_size': block_size
        }
        
        return quantized, stats
    
    def _calculate_quality_score(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Calculate quality score based on correlation and error metrics."""
        # Flatten tensors for correlation calculation
        orig_flat = original.flatten()
        recon_flat = reconstructed.flatten()
        
        # Normalize tensors
        orig_mean = orig_flat.mean()
        orig_std = orig_flat.std() + 1e-8
        recon_mean = recon_flat.mean()
        recon_std = recon_flat.std() + 1e-8
        
        orig_norm = (orig_flat - orig_mean) / orig_std
        recon_norm = (recon_flat - recon_mean) / recon_std
        
        # Calculate correlation coefficient manually to avoid dimension issues
        correlation = torch.mean(orig_norm * recon_norm)
        
        # Handle NaN correlation (constant tensors)
        if torch.isnan(correlation):
            correlation = torch.tensor(1.0)
        
        # Calculate relative error
        relative_error = torch.mean(torch.abs(orig_flat - recon_flat) / (torch.abs(orig_flat) + 1e-8))
        
        # Combine metrics (correlation is primary, error is penalty)
        quality_score = correlation.item() * (1 - min(relative_error.item(), 1.0))
        
        return max(0.0, min(1.0, quality_score))
    
    def should_quantize_tensor(self, name: str, tensor: torch.Tensor) -> bool:
        """Determine if a tensor should be quantized based on configuration."""
        # Check force list first (overrides everything)
        for force_pattern in self.config.force_layers:
            if force_pattern in name:
                return True
        
        # Check skip list
        for skip_pattern in self.config.skip_layers:
            if skip_pattern in name:
                return False
        
        # Default rules based on tensor properties
        if tensor.numel() < 1000:  # Skip very small tensors
            return False
        
        if 'bias' in name.lower():  # Skip bias terms by default
            return False
        
        if 'norm' in name.lower() or 'ln' in name.lower():  # Skip normalization layers
            return False
        
        return True    

    def quantize_model(self, model_state_dict: Dict[str, torch.Tensor]) -> QuantizationResult:
        """Quantize entire model and return comprehensive results."""
        start_time = time.time()
        
        total_tensors = len(model_state_dict)
        processed_tensors = 0
        
        original_size = 0
        quantized_size = 0
        tensor_results = {}
        quality_scores = []
        
        logger.info(f"Starting quantization with {self.config.quantization_type.value}")
        
        for name, tensor in model_state_dict.items():
            # Calculate original size
            tensor_original_size = tensor.numel() * 4  # F32 size
            original_size += tensor_original_size
            
            if self.should_quantize_tensor(name, tensor):
                # Quantize tensor
                quantized_data, stats = self.quantize_tensor(tensor, name)
                self.quantized_tensors[name] = quantized_data
                
                quantized_size += len(quantized_data)
                quality_scores.append(stats['quality_score'])
                
                # Check quality threshold
                if stats['quality_score'] < self.config.quality_threshold:
                    logger.warning(f"Low quality score for {name}: {stats['quality_score']:.3f}")
                
                tensor_results[name] = stats
                
            else:
                # Keep as F32
                f32_data = tensor.detach().cpu().numpy().astype(np.float32).tobytes()
                self.quantized_tensors[name] = np.frombuffer(f32_data, dtype=np.uint8)
                
                quantized_size += len(f32_data)
                quality_scores.append(1.0)  # Perfect quality for unquantized
                
                tensor_results[name] = {
                    'name': name,
                    'quantization_type': 'f32',
                    'quality_score': 1.0,
                    'original_size': tensor_original_size,
                    'quantized_size': len(f32_data),
                    'compression_ratio': 1.0,
                    'skipped': True
                }
            
            processed_tensors += 1
            progress = (processed_tensors / total_tensors) * 100
            self._report_progress(f"Quantized {name}", progress)
        
        # Calculate overall results
        processing_time = time.time() - start_time
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0
        overall_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        result = QuantizationResult(
            original_size=original_size,
            quantized_size=quantized_size,
            compression_ratio=compression_ratio,
            quality_score=overall_quality,
            processing_time=processing_time,
            tensor_results=tensor_results
        )
        
        logger.info(f"Quantization completed in {processing_time:.2f}s")
        logger.info(f"Compression ratio: {compression_ratio:.2f}x")
        logger.info(f"Overall quality score: {overall_quality:.3f}")
        logger.info(f"Size reduction: {result.size_reduction_mb:.2f} MB ({result.size_reduction_percent:.1f}%)")
        
        return result
    
    def validate_quantization(self, original_model: Dict[str, torch.Tensor], 
                            quantized_result: QuantizationResult) -> Dict[str, Any]:
        """Validate quantization quality and provide detailed analysis."""
        validation_results = {
            'overall_quality': quantized_result.quality_score,
            'compression_ratio': quantized_result.compression_ratio,
            'size_reduction_mb': quantized_result.size_reduction_mb,
            'low_quality_tensors': [],
            'high_compression_tensors': [],
            'validation_passed': True,
            'recommendations': []
        }
        
        # Analyze individual tensors
        for name, stats in quantized_result.tensor_results.items():
            if stats['quality_score'] < self.config.quality_threshold:
                validation_results['low_quality_tensors'].append({
                    'name': name,
                    'quality_score': stats['quality_score'],
                    'error_metrics': stats.get('error_metrics', {})
                })
                validation_results['validation_passed'] = False
            
            if stats.get('compression_ratio', 1.0) > 3.0:
                validation_results['high_compression_tensors'].append({
                    'name': name,
                    'compression_ratio': stats['compression_ratio'],
                    'quality_score': stats['quality_score']
                })
        
        # Generate recommendations
        if validation_results['low_quality_tensors']:
            validation_results['recommendations'].append(
                f"Consider adding {len(validation_results['low_quality_tensors'])} tensors to skip_layers"
            )
        
        if quantized_result.quality_score < 0.9:
            validation_results['recommendations'].append(
                "Overall quality is low - consider using higher precision quantization"
            )
        
        if quantized_result.compression_ratio < 1.5:
            validation_results['recommendations'].append(
                "Low compression ratio - consider more aggressive quantization"
            )
        
        return validation_results
    
    def save_quantization_report(self, result: QuantizationResult, 
                               validation: Dict[str, Any], 
                               output_path: Path) -> None:
        """Save comprehensive quantization report."""
        report = {
            'quantization_config': asdict(self.config),
            'quantization_result': asdict(result),
            'validation_result': validation,
            'tensor_details': result.tensor_results,
            'summary': {
                'total_tensors': len(result.tensor_results),
                'quantized_tensors': sum(1 for stats in result.tensor_results.values() 
                                       if not stats.get('skipped', False)),
                'skipped_tensors': sum(1 for stats in result.tensor_results.values() 
                                     if stats.get('skipped', False)),
                'average_quality': result.quality_score,
                'total_size_reduction_mb': result.size_reduction_mb,
                'compression_ratio': result.compression_ratio
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Quantization report saved to {output_path}")


def create_quantization_config(quantization_type: str, **kwargs) -> QuantizationConfig:
    """Create quantization configuration from string type."""
    try:
        quant_type = QuantizationType(quantization_type.lower())
    except ValueError:
        raise ValueError(f"Unsupported quantization type: {quantization_type}")
    
    return QuantizationConfig(quantization_type=quant_type, **kwargs)