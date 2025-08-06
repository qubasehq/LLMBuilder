"""
Dataset utilities for LLM training.
Provides efficient data loading and batching for tokenized text data.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from torch.utils.data import Dataset, DataLoader
from loguru import logger


class TextDataset(Dataset):
    """Dataset for tokenized text data with sliding window approach."""
    
    def __init__(self, 
                 data_path: Union[str, Path],
                 block_size: int = 256,
                 stride: Optional[int] = None):
        """
        Initialize text dataset.
        
        Args:
            data_path: Path to tokenized data file (.pt or .npy)
            block_size: Context window size
            stride: Stride for sliding window. If None, uses block_size (no overlap)
        """
        self.data_path = Path(data_path)
        self.block_size = block_size
        self.stride = stride or block_size
        
        # Load tokenized data
        self.tokens = self._load_tokens()
        
        # Calculate number of samples
        self.num_samples = max(0, (len(self.tokens) - block_size) // self.stride + 1)
        
        logger.info(f"Dataset initialized: {len(self.tokens):,} tokens, {self.num_samples:,} samples")
        logger.info(f"Block size: {block_size}, Stride: {self.stride}")
    
    def _load_tokens(self) -> torch.Tensor:
        """Load tokenized data from file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        try:
            if self.data_path.suffix == '.pt':
                tokens = torch.load(self.data_path, map_location='cpu')
            elif self.data_path.suffix == '.npy':
                tokens = torch.from_numpy(np.load(self.data_path))
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
            
            # Ensure tokens are long integers
            if tokens.dtype != torch.long:
                tokens = tokens.long()
            
            logger.info(f"Loaded {len(tokens):,} tokens from {self.data_path}")
            return tokens
            
        except Exception as e:
            logger.error(f"Error loading tokens: {e}")
            raise
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample.
        
        Returns:
            Dictionary with 'input_ids' and 'labels' tensors
        """
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")
        
        # Calculate start position
        start_idx = idx * self.stride
        end_idx = start_idx + self.block_size + 1  # +1 for target
        
        # Extract sequence
        sequence = self.tokens[start_idx:end_idx]
        
        # Handle edge case where sequence is shorter than expected
        if len(sequence) < self.block_size + 1:
            # Pad with zeros (or special padding token)
            padding_length = self.block_size + 1 - len(sequence)
            sequence = torch.cat([sequence, torch.zeros(padding_length, dtype=torch.long)])
        
        # Split into input and target
        input_ids = sequence[:-1]  # All but last token
        labels = sequence[1:]      # All but first token (shifted by 1)
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size from the data."""
        return int(self.tokens.max().item()) + 1


class MultiFileDataset(Dataset):
    """Dataset that can handle multiple tokenized files."""
    
    def __init__(self, 
                 data_dir: Union[str, Path],
                 block_size: int = 256,
                 stride: Optional[int] = None,
                 file_pattern: str = "*.pt"):
        """
        Initialize multi-file dataset.
        
        Args:
            data_dir: Directory containing tokenized files
            block_size: Context window size
            stride: Stride for sliding window
            file_pattern: Pattern to match data files
        """
        self.data_dir = Path(data_dir)
        self.block_size = block_size
        self.stride = stride or block_size
        
        # Find all data files
        self.data_files = list(self.data_dir.glob(file_pattern))
        
        if not self.data_files:
            raise FileNotFoundError(f"No data files found in {data_dir} matching {file_pattern}")
        
        # Load all data
        self.tokens = self._load_all_tokens()
        
        # Calculate number of samples
        self.num_samples = max(0, (len(self.tokens) - block_size) // self.stride + 1)
        
        logger.info(f"Multi-file dataset initialized: {len(self.data_files)} files")
        logger.info(f"Total tokens: {len(self.tokens):,}, Samples: {self.num_samples:,}")
    
    def _load_all_tokens(self) -> torch.Tensor:
        """Load and concatenate all tokenized files."""
        all_tokens = []
        
        for file_path in sorted(self.data_files):
            try:
                if file_path.suffix == '.pt':
                    tokens = torch.load(file_path, map_location='cpu')
                elif file_path.suffix == '.npy':
                    tokens = torch.from_numpy(np.load(file_path))
                else:
                    logger.warning(f"Skipping unsupported file: {file_path}")
                    continue
                
                if tokens.dtype != torch.long:
                    tokens = tokens.long()
                
                all_tokens.append(tokens)
                logger.info(f"Loaded {len(tokens):,} tokens from {file_path.name}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        if not all_tokens:
            raise ValueError("No valid token files could be loaded")
        
        # Concatenate all tokens
        combined_tokens = torch.cat(all_tokens, dim=0)
        logger.info(f"Combined {len(combined_tokens):,} total tokens")
        
        return combined_tokens
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample (same as TextDataset)."""
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")
        
        start_idx = idx * self.stride
        end_idx = start_idx + self.block_size + 1
        
        sequence = self.tokens[start_idx:end_idx]
        
        if len(sequence) < self.block_size + 1:
            padding_length = self.block_size + 1 - len(sequence)
            sequence = torch.cat([sequence, torch.zeros(padding_length, dtype=torch.long)])
        
        input_ids = sequence[:-1]
        labels = sequence[1:]
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size from the data."""
        return int(self.tokens.max().item()) + 1


def create_dataloader(dataset: Dataset, 
                     batch_size: int = 16,
                     shuffle: bool = True,
                     num_workers: int = 0,
                     pin_memory: bool = False) -> DataLoader:
    """Create a DataLoader with optimal settings for CPU training.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes (0 for single-threaded)
        pin_memory: Whether to pin memory (useful for GPU training)
        
    Returns:
        Configured DataLoader
    """
    # For CPU training, usually better to keep num_workers=0 to avoid overhead
    if num_workers > 0:
        logger.info(f"Using {num_workers} worker processes for data loading")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for consistent training
    )
    
    logger.info(f"DataLoader created: batch_size={batch_size}, shuffle={shuffle}")
    return dataloader


def split_dataset(dataset: Dataset, 
                 train_ratio: float = 0.9,
                 random_seed: int = 42) -> tuple:
    """Split dataset into train and validation sets.
    
    Args:
        dataset: Dataset to split
        train_ratio: Ratio of data for training
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    from torch.utils.data import random_split
    
    # Set random seed for reproducibility
    generator = torch.Generator().manual_seed(random_seed)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    logger.info(f"Dataset split: {train_size:,} train, {val_size:,} validation")
    return train_dataset, val_dataset


def estimate_dataset_memory(dataset: Dataset) -> dict:
    """Estimate memory usage of dataset.
    
    Args:
        dataset: Dataset to analyze
        
    Returns:
        Dictionary with memory estimates in MB
    """
    if hasattr(dataset, 'tokens'):
        # For TextDataset and MultiFileDataset
        token_memory = dataset.tokens.numel() * dataset.tokens.element_size() / (1024 * 1024)
        return {
            'tokens_mb': token_memory,
            'estimated_total_mb': token_memory * 1.2  # Add 20% overhead
        }
    else:
        # For other datasets, rough estimate
        sample = dataset[0]
        sample_size = sum(tensor.numel() * tensor.element_size() for tensor in sample.values())
        total_memory = sample_size * len(dataset) / (1024 * 1024)
        return {
            'estimated_total_mb': total_memory
        }

