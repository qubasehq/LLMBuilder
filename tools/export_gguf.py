"""
Enhanced GGUF export utility for LLM models.
Converts trained PyTorch models to GGUF format for llama.cpp compatibility.
Features improved metadata handling, model validation, and comprehensive tensor writing.
"""

import os
import struct
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import time
from loguru import logger


@dataclass
class ModelMetadata:
    """Enhanced model metadata for GGUF export."""
    name: str
    architecture: str
    version: str
    author: Optional[str] = None
    description: Optional[str] = None
    license: Optional[str] = None
    url: Optional[str] = None
    
    # Training metadata
    training_data: Optional[str] = None
    training_steps: Optional[int] = None
    training_time: Optional[float] = None
    loss: Optional[float] = None
    
    # Model parameters
    vocab_size: Optional[int] = None
    context_length: Optional[int] = None
    embedding_length: Optional[int] = None
    block_count: Optional[int] = None
    head_count: Optional[int] = None
    
    # Export metadata
    export_timestamp: Optional[str] = None
    export_version: str = "1.0"
    source_model_hash: Optional[str] = None
    
    def to_gguf_metadata(self) -> Dict[str, Any]:
        """Convert to GGUF metadata format."""
        metadata = {}
        
        # General metadata
        if self.name:
            metadata["general.name"] = self.name
        if self.architecture:
            metadata["general.architecture"] = self.architecture
        if self.version:
            metadata["general.version"] = self.version
        if self.author:
            metadata["general.author"] = self.author
        if self.description:
            metadata["general.description"] = self.description
        if self.license:
            metadata["general.license"] = self.license
        if self.url:
            metadata["general.url"] = self.url
            
        # Training metadata
        if self.training_data:
            metadata["training.data"] = self.training_data
        if self.training_steps:
            metadata["training.steps"] = self.training_steps
        if self.training_time:
            metadata["training.time"] = self.training_time
        if self.loss:
            metadata["training.loss"] = self.loss
            
        # Export metadata
        if self.export_timestamp:
            metadata["export.timestamp"] = self.export_timestamp
        if self.export_version:
            metadata["export.version"] = self.export_version
        if self.source_model_hash:
            metadata["export.source_hash"] = self.source_model_hash
            
        return metadata


@dataclass
class GGUFTypedValue:
    """Wrapper for values that need specific GGUF types."""
    value: Any
    gguf_type: int

@dataclass
class GGUFTensorInfo:
    """Information about a tensor in GGUF format."""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    size_bytes: int
    quantization: Optional[str] = None
    
    
@dataclass
class GGUFValidationResult:
    """Result of GGUF validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    file_size: int
    tensor_count: int
    metadata_count: int


class GGUFConverter:
    """Enhanced GGUF converter with improved metadata handling and validation."""
    
    # GGUF data types
    GGUF_TYPE_UINT8 = 0
    GGUF_TYPE_INT8 = 1
    GGUF_TYPE_UINT16 = 2
    GGUF_TYPE_INT16 = 3
    GGUF_TYPE_UINT32 = 4
    GGUF_TYPE_INT32 = 5
    GGUF_TYPE_FLOAT32 = 6
    GGUF_TYPE_BOOL = 7
    GGUF_TYPE_STRING = 8
    GGUF_TYPE_ARRAY = 9
    GGUF_TYPE_UINT64 = 10
    GGUF_TYPE_INT64 = 11
    GGUF_TYPE_FLOAT64 = 12
    
    def __init__(self, model_path: str, output_path: str, metadata: Optional[ModelMetadata] = None, quantization_type: str = "f16"):
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.quantization_type = quantization_type
        self.metadata = metadata or ModelMetadata(name="Unknown", architecture="gpt2", version="1.0")
        self.model = None
        self.config = None
        self.tokenizer_vocab = None
        self.tensor_info: List[GGUFTensorInfo] = []
        
    def load_model(self) -> bool:
        """Load PyTorch model from checkpoint with enhanced metadata extraction."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Extract model and config
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    self.model = checkpoint['model']
                    self.config = checkpoint.get('config', {})
                elif 'model_state_dict' in checkpoint:
                    self.model = checkpoint['model_state_dict']
                    self.config = checkpoint.get('config', {})
                elif 'state_dict' in checkpoint:
                    self.model = checkpoint['state_dict']
                    self.config = checkpoint.get('config', {})
                else:
                    # Assume the checkpoint is the model state dict
                    self.model = checkpoint
                    self.config = {}
            else:
                # Direct model object
                self.model = checkpoint
                self.config = getattr(checkpoint, 'config', {})
            
            # Extract additional metadata from checkpoint
            if isinstance(checkpoint, dict):
                self._extract_training_metadata(checkpoint)
            
            # Calculate model hash for validation
            self.metadata.source_model_hash = self._calculate_model_hash()
            self.metadata.export_timestamp = datetime.now().isoformat()
            
            # Update metadata with model architecture info
            self._update_architecture_metadata()
            
            logger.info(f"Successfully loaded model: {self.metadata.name}")
            logger.info(f"Architecture: {self.metadata.architecture}")
            logger.info(f"Parameters: {self._count_parameters():,}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_training_metadata(self, checkpoint: Dict[str, Any]) -> None:
        """Extract training metadata from checkpoint."""
        if 'training_stats' in checkpoint:
            stats = checkpoint['training_stats']
            self.metadata.training_steps = stats.get('step', stats.get('steps'))
            self.metadata.training_time = stats.get('training_time')
            self.metadata.loss = stats.get('loss', stats.get('final_loss'))
        
        # Also check for step/epoch at top level
        if 'step' in checkpoint:
            self.metadata.training_steps = checkpoint['step']
        elif 'global_step' in checkpoint:
            self.metadata.training_steps = checkpoint['global_step']
        
        if 'epoch' in checkpoint and not self.metadata.training_steps:
            # If no step info, use epoch
            self.metadata.training_steps = checkpoint['epoch']
        
        if 'loss' in checkpoint and not self.metadata.loss:
            self.metadata.loss = checkpoint['loss']
        
        if 'optimizer' in checkpoint:
            # Model was saved during training
            pass
    
    def _calculate_model_hash(self) -> str:
        """Calculate SHA256 hash of model parameters."""
        hasher = hashlib.sha256()
        
        if isinstance(self.model, dict):
            # State dict
            for key in sorted(self.model.keys()):
                tensor = self.model[key]
                if isinstance(tensor, torch.Tensor):
                    hasher.update(tensor.detach().cpu().numpy().tobytes())
        else:
            # Model object
            for param in self.model.parameters():
                hasher.update(param.detach().cpu().numpy().tobytes())
        
        return hasher.hexdigest()
    
    def _update_architecture_metadata(self) -> None:
        """Update metadata with architecture-specific information."""
        if isinstance(self.config, dict):
            self.metadata.vocab_size = self.config.get('vocab_size')
            self.metadata.context_length = self.config.get('block_size', self.config.get('max_position_embeddings'))
            self.metadata.embedding_length = self.config.get('n_embd', self.config.get('hidden_size'))
            self.metadata.block_count = self.config.get('n_layer', self.config.get('num_hidden_layers'))
            self.metadata.head_count = self.config.get('n_head', self.config.get('num_attention_heads'))
    
    def _count_parameters(self) -> int:
        """Count total number of parameters in the model."""
        if isinstance(self.model, dict):
            return sum(tensor.numel() for tensor in self.model.values() if isinstance(tensor, torch.Tensor))
        else:
            return sum(p.numel() for p in self.model.parameters())
    
    def load_tokenizer_vocab(self, tokenizer_path: Optional[str] = None) -> bool:
        """Load tokenizer vocabulary for enhanced metadata."""
        try:
            if tokenizer_path:
                vocab_path = Path(tokenizer_path)
            else:
                # Try to find tokenizer in common locations
                possible_paths = [
                    self.model_path.parent / "tokenizer.json",
                    self.model_path.parent / "vocab.json", 
                    self.model_path.parent / "tokenizer" / "tokenizer.json",
                    Path("tokenizer") / "tokenizer.json"
                ]
                
                vocab_path = None
                for path in possible_paths:
                    if path.exists():
                        vocab_path = path
                        break
            
            if vocab_path and vocab_path.exists():
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    self.tokenizer_vocab = json.load(f)
                logger.info(f"Loaded tokenizer vocabulary from {vocab_path}")
                return True
            else:
                logger.warning("No tokenizer vocabulary found - using default tokens")
                return False
                
        except Exception as e:
            logger.warning(f"Failed to load tokenizer vocabulary: {e}")
            return False
    
    def create_gguf_metadata(self) -> Dict[str, Any]:
        """Create comprehensive GGUF metadata from model config and metadata."""
        # Start with custom metadata
        metadata = self.metadata.to_gguf_metadata()
        
        # Add standard GGUF metadata
        metadata.update({
            "general.quantization_version": GGUFTypedValue(2, self.GGUF_TYPE_UINT32),
            "general.file_type": GGUFTypedValue(1, self.GGUF_TYPE_UINT32),  # F32
        })
        
        # Architecture-specific metadata
        arch = self.metadata.architecture.lower()
        if arch in ["gpt2", "gpt"]:
            self._add_gpt2_metadata(metadata)
        elif arch in ["llama", "llama2"]:
            self._add_llama_metadata(metadata)
        elif arch in ["bert"]:
            self._add_bert_metadata(metadata)
        else:
            # Default to GPT2-like architecture
            self._add_gpt2_metadata(metadata)
        
        # Add tokenizer metadata if available
        self._add_tokenizer_metadata(metadata)
        
        return metadata
    
    def _add_gpt2_metadata(self, metadata: Dict[str, Any]) -> None:
        """Add GPT2-specific metadata."""
        prefix = "gpt2"
        
        metadata[f"{prefix}.context_length"] = GGUFTypedValue(self.metadata.context_length or self.config.get('block_size', 1024), self.GGUF_TYPE_UINT32)
        metadata[f"{prefix}.embedding_length"] = GGUFTypedValue(self.metadata.embedding_length or self.config.get('n_embd', 768), self.GGUF_TYPE_UINT32)
        metadata[f"{prefix}.feed_forward_length"] = GGUFTypedValue((self.metadata.embedding_length or self.config.get('n_embd', 768)) * 4, self.GGUF_TYPE_UINT32)
        metadata[f"{prefix}.block_count"] = GGUFTypedValue(self.metadata.block_count or self.config.get('n_layer', 12), self.GGUF_TYPE_UINT32)
        metadata[f"{prefix}.attention.head_count"] = GGUFTypedValue(self.metadata.head_count or self.config.get('n_head', 12), self.GGUF_TYPE_UINT32)
        metadata[f"{prefix}.attention.head_count_kv"] = GGUFTypedValue(self.metadata.head_count or self.config.get('n_head', 12), self.GGUF_TYPE_UINT32)
        metadata[f"{prefix}.attention.layer_norm_epsilon"] = self.config.get('layer_norm_epsilon', 1e-5)
        metadata[f"{prefix}.vocab_size"] = GGUFTypedValue(self.metadata.vocab_size or self.config.get('vocab_size', 50257), self.GGUF_TYPE_UINT32)
        
        # Calculate rope dimension
        embd_dim = self.metadata.embedding_length or self.config.get('n_embd', 768)
        head_count = self.metadata.head_count or self.config.get('n_head', 12)
        metadata[f"{prefix}.rope.dimension_count"] = GGUFTypedValue(embd_dim // head_count, self.GGUF_TYPE_UINT32)
    
    def _add_llama_metadata(self, metadata: Dict[str, Any]) -> None:
        """Add LLaMA-specific metadata."""
        prefix = "llama"
        
        metadata[f"{prefix}.context_length"] = self.metadata.context_length or self.config.get('max_position_embeddings', 2048)
        metadata[f"{prefix}.embedding_length"] = self.metadata.embedding_length or self.config.get('hidden_size', 4096)
        metadata[f"{prefix}.feed_forward_length"] = self.config.get('intermediate_size', 11008)
        metadata[f"{prefix}.block_count"] = self.metadata.block_count or self.config.get('num_hidden_layers', 32)
        metadata[f"{prefix}.attention.head_count"] = self.metadata.head_count or self.config.get('num_attention_heads', 32)
        metadata[f"{prefix}.attention.head_count_kv"] = self.config.get('num_key_value_heads', metadata[f"{prefix}.attention.head_count"])
        metadata[f"{prefix}.attention.layer_norm_rms_epsilon"] = self.config.get('rms_norm_eps', 1e-6)
        metadata[f"{prefix}.vocab_size"] = self.metadata.vocab_size or self.config.get('vocab_size', 32000)
        metadata[f"{prefix}.rope.dimension_count"] = self.config.get('rope_theta', 10000.0)
    
    def _add_bert_metadata(self, metadata: Dict[str, Any]) -> None:
        """Add BERT-specific metadata."""
        prefix = "bert"
        
        metadata[f"{prefix}.context_length"] = self.metadata.context_length or self.config.get('max_position_embeddings', 512)
        metadata[f"{prefix}.embedding_length"] = self.metadata.embedding_length or self.config.get('hidden_size', 768)
        metadata[f"{prefix}.feed_forward_length"] = self.config.get('intermediate_size', 3072)
        metadata[f"{prefix}.block_count"] = self.metadata.block_count or self.config.get('num_hidden_layers', 12)
        metadata[f"{prefix}.attention.head_count"] = self.metadata.head_count or self.config.get('num_attention_heads', 12)
        metadata[f"{prefix}.attention.layer_norm_epsilon"] = self.config.get('layer_norm_eps', 1e-12)
        metadata[f"{prefix}.vocab_size"] = self.metadata.vocab_size or self.config.get('vocab_size', 30522)
    
    def _add_tokenizer_metadata(self, metadata: Dict[str, Any]) -> None:
        """Add tokenizer metadata if available."""
        if self.tokenizer_vocab:
            if isinstance(self.tokenizer_vocab, dict):
                if 'model' in self.tokenizer_vocab:
                    metadata["tokenizer.ggml.model"] = self.tokenizer_vocab['model'].get('type', 'bpe')
                
                if 'vocab' in self.tokenizer_vocab:
                    vocab = self.tokenizer_vocab['vocab']
                    metadata["tokenizer.ggml.tokens"] = list(vocab.keys())
                    metadata["tokenizer.ggml.scores"] = [0.0] * len(vocab)  # Default scores
                
                # Add special token IDs
                special_tokens = self.tokenizer_vocab.get('added_tokens', [])
                for token_info in special_tokens:
                    if isinstance(token_info, dict):
                        content = token_info.get('content', '')
                        token_id = token_info.get('id', 0)
                        
                        if content in ['<s>', '[CLS]']:
                            metadata["tokenizer.ggml.bos_token_id"] = GGUFTypedValue(token_id, self.GGUF_TYPE_UINT32)
                        elif content in ['</s>', '[SEP]']:
                            metadata["tokenizer.ggml.eos_token_id"] = GGUFTypedValue(token_id, self.GGUF_TYPE_UINT32)
                        elif content in ['<unk>', '[UNK]']:
                            metadata["tokenizer.ggml.unk_token_id"] = GGUFTypedValue(token_id, self.GGUF_TYPE_UINT32)
                        elif content in ['<pad>', '[PAD]']:
                            metadata["tokenizer.ggml.padding_token_id"] = GGUFTypedValue(token_id, self.GGUF_TYPE_UINT32)
        else:
            # Default tokenizer metadata
            metadata.update({
                "tokenizer.ggml.model": "gpt2",
                "tokenizer.ggml.tokens": [],
                "tokenizer.ggml.scores": [],
                "tokenizer.ggml.merges": [],
                "tokenizer.ggml.bos_token_id": GGUFTypedValue(1, self.GGUF_TYPE_UINT32),
                "tokenizer.ggml.eos_token_id": GGUFTypedValue(2, self.GGUF_TYPE_UINT32),
                "tokenizer.ggml.unk_token_id": GGUFTypedValue(0, self.GGUF_TYPE_UINT32),
                "tokenizer.ggml.padding_token_id": GGUFTypedValue(0, self.GGUF_TYPE_UINT32),
            })
    
    def export_to_gguf(self, tokenizer_path: Optional[str] = None, validate: bool = True) -> bool:
        """Export model to GGUF format with comprehensive metadata and validation."""
        try:
            logger.info("Starting GGUF export process...")
            start_time = time.time()
            
            # Load model and tokenizer
            if not self.load_model():
                return False
            
            self.load_tokenizer_vocab(tokenizer_path)
            
            # Prepare output directory
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create GGUF file
            with open(self.output_path, 'wb') as f:
                # Collect tensor information first
                self._collect_tensor_info()
                
                # Write GGUF header
                metadata = self.create_gguf_metadata()
                self._write_header(f, len(self.tensor_info), len(metadata))
                
                # Write metadata
                self._write_metadata(f, metadata)
                
                # Write tensor data
                self._write_tensors(f)
            
            export_time = time.time() - start_time
            
            # Validate the exported file
            if validate:
                validation_result = self.validate_gguf_file()
                if not validation_result.is_valid:
                    logger.error("GGUF validation failed:")
                    for error in validation_result.errors:
                        logger.error(f"  - {error}")
                    return False
                
                if validation_result.warnings:
                    for warning in validation_result.warnings:
                        logger.warning(f"  - {warning}")
            
            # Log export statistics
            file_size = self.output_path.stat().st_size
            logger.info(f"Successfully exported to {self.output_path}")
            logger.info(f"Export time: {export_time:.2f}s")
            logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
            logger.info(f"Tensors: {len(self.tensor_info)}")
            logger.info(f"Metadata entries: {len(metadata)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to export to GGUF: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _collect_tensor_info(self) -> None:
        """Collect information about all tensors before writing."""
        self.tensor_info = []
        
        if isinstance(self.model, dict):
            state_dict = self.model
        else:
            state_dict = self.model.state_dict()
        
        # Get quantization block size requirements
        block_size = self._get_quantization_block_size()
        
        for name, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                # Check tensor alignment for quantization
                original_tensor = tensor.detach().cpu()
                aligned_tensor = self._ensure_tensor_alignment(original_tensor, block_size, name)
                
                np_tensor = aligned_tensor.numpy()
                
                tensor_info = GGUFTensorInfo(
                    name=name,
                    shape=tuple(np_tensor.shape),
                    dtype=str(np_tensor.dtype),
                    size_bytes=np_tensor.nbytes
                )
                
                self.tensor_info.append(tensor_info)
        
        logger.info(f"Collected information for {len(self.tensor_info)} tensors")
    
    def _get_quantization_block_size(self) -> int:
        """Get the block size requirement for the current quantization type."""
        # Most quantization types require multiples of 32
        quantization_block_sizes = {
            'f16': 1,
            'f32': 1,
            'q4_0': 32,
            'q4_1': 32,
            'q5_0': 32,
            'q5_1': 32,
            'q8_0': 32,
            'q8_1': 32,
        }
        return quantization_block_sizes.get(self.quantization_type.lower(), 32)
    
    def _ensure_tensor_alignment(self, tensor: torch.Tensor, block_size: int, tensor_name: str) -> torch.Tensor:
        """Ensure tensor dimensions are compatible with quantization block size."""
        if block_size == 1:
            return tensor  # No alignment needed for f16/f32
        
        shape = list(tensor.shape)
        needs_padding = False
        
        # Check each dimension for alignment
        for i, dim in enumerate(shape):
            if dim % block_size != 0:
                new_dim = ((dim + block_size - 1) // block_size) * block_size
                logger.warning(f"Tensor '{tensor_name}' dimension {i} ({dim}) not aligned to block size {block_size}, padding to {new_dim}")
                shape[i] = new_dim
                needs_padding = True
        
        if needs_padding:
            # Create new tensor with aligned dimensions
            aligned_tensor = torch.zeros(shape, dtype=tensor.dtype)
            
            # Copy original data (truncate if necessary)
            slices = []
            for i, (orig_dim, new_dim) in enumerate(zip(tensor.shape, shape)):
                slices.append(slice(0, min(orig_dim, new_dim)))
            
            aligned_tensor[tuple(slices)] = tensor[tuple(slices)]
            return aligned_tensor
        
        return tensor
    
    def _write_header(self, f, tensor_count: int, metadata_count: int) -> None:
        """Write comprehensive GGUF header."""
        # GGUF magic number (4 bytes) + version (4 bytes)
        magic = b'GGUF'
        f.write(magic)
        
        # Version (3 for current GGUF spec)
        f.write(struct.pack('<I', 3))
        
        # Tensor count (8 bytes)
        f.write(struct.pack('<Q', tensor_count))
        
        # Metadata key-value count (8 bytes)
        f.write(struct.pack('<Q', metadata_count))
        
        logger.debug(f"Written GGUF header: {tensor_count} tensors, {metadata_count} metadata entries")
    
    def _write_metadata(self, f, metadata: Dict[str, Any]) -> None:
        """Write comprehensive metadata key-value pairs according to GGUF spec."""
        for key, value in metadata.items():
            # Write key
            key_bytes = key.encode('utf-8')
            f.write(struct.pack('<Q', len(key_bytes)))
            f.write(key_bytes)
            
            # Write value based on type
            if isinstance(value, GGUFTypedValue):
                # Explicitly typed value
                f.write(struct.pack('<I', value.gguf_type))
                if value.gguf_type == self.GGUF_TYPE_UINT32:
                    f.write(struct.pack('<I', value.value))
                elif value.gguf_type == self.GGUF_TYPE_INT32:
                    f.write(struct.pack('<i', value.value))
                elif value.gguf_type == self.GGUF_TYPE_UINT64:
                    f.write(struct.pack('<Q', value.value))
                elif value.gguf_type == self.GGUF_TYPE_INT64:
                    f.write(struct.pack('<q', value.value))
                elif value.gguf_type == self.GGUF_TYPE_FLOAT32:
                    f.write(struct.pack('<f', value.value))
                elif value.gguf_type == self.GGUF_TYPE_FLOAT64:
                    f.write(struct.pack('<d', value.value))
                else:
                    raise ValueError(f"Unsupported GGUF type: {value.gguf_type}")
                    
            elif isinstance(value, str):
                # String type
                f.write(struct.pack('<I', self.GGUF_TYPE_STRING))
                value_bytes = value.encode('utf-8')
                f.write(struct.pack('<Q', len(value_bytes)))
                f.write(value_bytes)
                
            elif isinstance(value, bool):
                # Boolean type
                f.write(struct.pack('<I', self.GGUF_TYPE_BOOL))
                f.write(struct.pack('<?', value))
                
            elif isinstance(value, int):
                # Integer type - choose appropriate size
                if -2**31 <= value < 2**31:
                    f.write(struct.pack('<I', self.GGUF_TYPE_INT32))
                    f.write(struct.pack('<i', value))
                else:
                    f.write(struct.pack('<I', self.GGUF_TYPE_INT64))
                    f.write(struct.pack('<q', value))
                    
            elif isinstance(value, float):
                # Float type
                f.write(struct.pack('<I', self.GGUF_TYPE_FLOAT32))
                f.write(struct.pack('<f', value))
                
            elif isinstance(value, list):
                # Array type
                f.write(struct.pack('<I', self.GGUF_TYPE_ARRAY))
                
                if len(value) > 0:
                    # Determine array element type
                    first_elem = value[0]
                    if isinstance(first_elem, str):
                        elem_type = self.GGUF_TYPE_STRING
                    elif isinstance(first_elem, int):
                        elem_type = self.GGUF_TYPE_INT32
                    elif isinstance(first_elem, float):
                        elem_type = self.GGUF_TYPE_FLOAT32
                    else:
                        elem_type = self.GGUF_TYPE_STRING  # Default
                        
                    f.write(struct.pack('<I', elem_type))
                    f.write(struct.pack('<Q', len(value)))
                    
                    # Write array elements
                    for elem in value:
                        if elem_type == self.GGUF_TYPE_STRING:
                            elem_bytes = str(elem).encode('utf-8')
                            f.write(struct.pack('<Q', len(elem_bytes)))
                            f.write(elem_bytes)
                        elif elem_type == self.GGUF_TYPE_INT32:
                            f.write(struct.pack('<i', int(elem)))
                        elif elem_type == self.GGUF_TYPE_FLOAT32:
                            f.write(struct.pack('<f', float(elem)))
                else:
                    # Empty array
                    f.write(struct.pack('<I', self.GGUF_TYPE_STRING))
                    f.write(struct.pack('<Q', 0))
            else:
                # Fallback to string representation
                f.write(struct.pack('<I', self.GGUF_TYPE_STRING))
                value_bytes = str(value).encode('utf-8')
                f.write(struct.pack('<Q', len(value_bytes)))
                f.write(value_bytes)
        
        logger.debug(f"Written {len(metadata)} metadata entries")
    
    def _write_tensors(self, f) -> None:
        """Write comprehensive tensor data according to GGUF spec."""
        if isinstance(self.model, dict):
            state_dict = self.model
        else:
            state_dict = self.model.state_dict()
        
        tensor_count = 0
        total_size = 0
        
        for tensor_info in self.tensor_info:
            tensor = state_dict[tensor_info.name]
            np_tensor = tensor.detach().cpu().numpy()
            
            # Write tensor name
            tensor_name = tensor_info.name.encode('utf-8')
            f.write(struct.pack('<Q', len(tensor_name)))
            f.write(tensor_name)
            
            # Write number of dimensions
            f.write(struct.pack('<I', len(tensor_info.shape)))
            
            # Write dimensions
            for dim in tensor_info.shape:
                f.write(struct.pack('<Q', dim))
            
            # Write tensor type (F32 for now)
            f.write(struct.pack('<I', self.GGUF_TYPE_FLOAT32))
            
            # Write tensor offset (will be calculated during actual writing)
            offset_pos = f.tell()
            f.write(struct.pack('<Q', 0))  # Placeholder for offset
            
            tensor_count += 1
            total_size += tensor_info.size_bytes
        
        # Align to 32-byte boundary for tensor data
        current_pos = f.tell()
        alignment = 32
        padding = (alignment - (current_pos % alignment)) % alignment
        f.write(b'\x00' * padding)
        
        # Now write actual tensor data and update offsets
        tensor_data_start = f.tell()
        current_offset = 0
        
        for i, tensor_info in enumerate(self.tensor_info):
            tensor = state_dict[tensor_info.name]
            np_tensor = tensor.detach().cpu().numpy().astype(np.float32)
            
            # Write tensor data
            tensor_bytes = np_tensor.tobytes()
            f.write(tensor_bytes)
            
            # Update offset in header (go back and write the actual offset)
            current_pos = f.tell()
            offset_position = tensor_data_start - (len(self.tensor_info) - i) * 8
            f.seek(offset_position)
            f.write(struct.pack('<Q', tensor_data_start + current_offset))
            f.seek(current_pos)
            
            current_offset += len(tensor_bytes)
            
            # Align each tensor to 32-byte boundary
            padding = (alignment - (len(tensor_bytes) % alignment)) % alignment
            if padding > 0:
                f.write(b'\x00' * padding)
                current_offset += padding
        
        logger.info(f"Written {tensor_count} tensors, total size: {total_size / (1024*1024):.2f} MB")


    def validate_gguf_file(self) -> GGUFValidationResult:
        """Validate the exported GGUF file for correctness."""
        errors = []
        warnings = []
        
        try:
            if not self.output_path.exists():
                errors.append("Output file does not exist")
                return GGUFValidationResult(False, errors, warnings, 0, 0, 0)
            
            file_size = self.output_path.stat().st_size
            
            with open(self.output_path, 'rb') as f:
                # Check magic number
                magic = f.read(4)
                if magic != b'GGUF':
                    errors.append(f"Invalid magic number: {magic}")
                
                # Check version
                version = struct.unpack('<I', f.read(4))[0]
                if version != 3:
                    warnings.append(f"Unexpected version: {version} (expected 3)")
                
                # Read counts
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                metadata_count = struct.unpack('<Q', f.read(8))[0]
                
                # Validate counts
                if tensor_count != len(self.tensor_info):
                    errors.append(f"Tensor count mismatch: file={tensor_count}, expected={len(self.tensor_info)}")
                
                if tensor_count == 0:
                    errors.append("No tensors found in file")
                
                if metadata_count == 0:
                    warnings.append("No metadata found in file")
                
                # Basic file size validation
                min_expected_size = 20 + metadata_count * 16 + tensor_count * 32  # Very rough estimate
                if file_size < min_expected_size:
                    errors.append(f"File size too small: {file_size} bytes")
                
                # Check if file is complete (not truncated)
                f.seek(0, 2)  # Seek to end
                actual_size = f.tell()
                if actual_size != file_size:
                    errors.append("File appears to be truncated")
            
            is_valid = len(errors) == 0
            
            if is_valid:
                logger.info("GGUF file validation passed")
            else:
                logger.error("GGUF file validation failed")
            
            return GGUFValidationResult(is_valid, errors, warnings, file_size, tensor_count, metadata_count)
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
            return GGUFValidationResult(False, errors, warnings, 0, 0, 0)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information for reporting."""
        if not self.model:
            return {}
        
        info = {
            "metadata": asdict(self.metadata),
            "config": self.config,
            "parameter_count": self._count_parameters(),
            "tensor_count": len(self.tensor_info),
            "tensors": [asdict(t) for t in self.tensor_info],
        }
        
        if self.output_path.exists():
            info["output_file"] = {
                "path": str(self.output_path),
                "size_bytes": self.output_path.stat().st_size,
                "size_mb": self.output_path.stat().st_size / (1024 * 1024)
            }
        
        return info


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


def export_to_gguf_cli(input_path: str, output_path: str, quantization: str = "f32", 
                      tokenizer_path: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                      validate: bool = True) -> bool:
    """Enhanced CLI function for GGUF export with comprehensive options."""
    try:
        logger.info(f"Starting GGUF export: {input_path} -> {output_path}")
        
        # Create metadata object
        model_metadata = ModelMetadata(
            name=metadata.get('name', 'LLMBuilder Model') if metadata else 'LLMBuilder Model',
            architecture=metadata.get('architecture', 'gpt2') if metadata else 'gpt2',
            version=metadata.get('version', '1.0') if metadata else '1.0',
            author=metadata.get('author') if metadata else None,
            description=metadata.get('description') if metadata else None,
            license=metadata.get('license') if metadata else None,
        )
        
        # Create converter
        converter = GGUFConverter(input_path, output_path, model_metadata, quantization)
        
        if quantization != "f32":
            quantizer = GGMLQuantizer(input_path)
            logger.info(f"Using {quantization} quantization")
            # TODO: Integrate quantization with converter
            
        # Export model
        success = converter.export_to_gguf(tokenizer_path, validate)
        
        if success:
            # Get comprehensive model info
            model_info = converter.get_model_info()
            
            # Calculate file sizes
            input_size = Path(input_path).stat().st_size / (1024 * 1024)
            output_size = Path(output_path).stat().st_size / (1024 * 1024)
            
            logger.info("=== Export Summary ===")
            logger.info(f"Model: {model_metadata.name}")
            logger.info(f"Architecture: {model_metadata.architecture}")
            logger.info(f"Parameters: {model_info.get('parameter_count', 0):,}")
            logger.info(f"Tensors: {model_info.get('tensor_count', 0)}")
            logger.info(f"Input size: {input_size:.2f} MB")
            logger.info(f"Output size: {output_size:.2f} MB")
            logger.info(f"Compression ratio: {input_size/output_size:.2f}x")
            
            # Save model info to JSON
            info_path = Path(output_path).with_suffix('.json')
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, default=str)
            logger.info(f"Model info saved to: {info_path}")
            
        return success
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_model_metadata_from_args(args) -> ModelMetadata:
    """Create ModelMetadata from command line arguments."""
    return ModelMetadata(
        name=getattr(args, 'name', 'LLMBuilder Model'),
        architecture=getattr(args, 'architecture', 'gpt2'),
        version=getattr(args, 'version', '1.0'),
        author=getattr(args, 'author', None),
        description=getattr(args, 'description', None),
        license=getattr(args, 'license', None),
        url=getattr(args, 'url', None),
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced GGUF export utility with comprehensive metadata handling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic export
  python export_gguf.py --input model.pt --output model.gguf
  
  # Export with metadata
  python export_gguf.py --input model.pt --output model.gguf \\
    --name "My Model" --architecture gpt2 --author "Me"
  
  # Export with tokenizer
  python export_gguf.py --input model.pt --output model.gguf \\
    --tokenizer tokenizer.json
  
  # Export with quantization (future)
  python export_gguf.py --input model.pt --output model.gguf \\
    --quantization q4_0
        """
    )
    
    # Required arguments
    parser.add_argument("--input", required=True, help="Path to input model checkpoint")
    parser.add_argument("--output", required=True, help="Output GGUF file path")
    
    # Model metadata
    parser.add_argument("--name", help="Model name")
    parser.add_argument("--architecture", choices=["gpt2", "gpt", "llama", "llama2", "bert"], 
                       default="gpt2", help="Model architecture")
    parser.add_argument("--version", default="1.0", help="Model version")
    parser.add_argument("--author", help="Model author")
    parser.add_argument("--description", help="Model description")
    parser.add_argument("--license", help="Model license")
    parser.add_argument("--url", help="Model URL")
    
    # Export options
    parser.add_argument("--tokenizer", help="Path to tokenizer vocabulary file")
    parser.add_argument("--quantization", choices=["f32", "f16", "q8_0", "q4_0"], 
                       default="f32", help="Quantization format")
    parser.add_argument("--no-validate", action="store_true", 
                       help="Skip validation of exported file")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(lambda msg: print(msg, end=''), level="DEBUG")
    
    # Create metadata
    metadata = {
        'name': args.name,
        'architecture': args.architecture,
        'version': args.version,
        'author': args.author,
        'description': args.description,
        'license': args.license,
        'url': args.url,
    }
    
    # Export model
    success = export_to_gguf_cli(
        args.input, 
        args.output, 
        args.quantization,
        args.tokenizer,
        metadata,
        not args.no_validate
    )
    
    exit(0 if success else 1)
