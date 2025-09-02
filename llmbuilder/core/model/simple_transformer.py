#!/usr/bin/env python3
"""
Simple but complete transformer model for text generation.
Compatible with our inference system and training pipeline.
"""

import math
from llmbuilder.utils.lazy_imports import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        # Query, Key, Value projections
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence, embedding
        
        # Calculate Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


class FeedForward(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, n_embd: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block."""
    
    def __init__(self, n_embd: int, n_head: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd, hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class SimpleTransformer(nn.Module):
    """Simple but complete transformer model for text generation."""
    
    def __init__(self, 
                 vocab_size: int,
                 n_embd: int = 384,
                 n_layer: int = 6,
                 n_head: int = 6,
                 block_size: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        # Store configuration
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.block_size = block_size
        self.dropout = dropout
        
        # Validate configuration
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        
        # Transformer blocks
        hidden_dim = 4 * n_embd  # Standard transformer ratio
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, hidden_dim, dropout)
            for _ in range(n_layer)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Create causal mask
        self.register_buffer("causal_mask", 
                           torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        
        print(f"SimpleTransformer initialized with {self.get_num_params():,} parameters")
        print(f"Architecture: {n_layer} layers, {n_head} heads, {n_embd} embedding dim")
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"
        
        # Token embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        
        # Position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, T)
        pos_emb = self.position_embedding(pos)  # (1, T, n_embd)
        
        # Combine embeddings
        x = tok_emb + pos_emb
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply transformer blocks
        mask = self.causal_mask[:, :, :T, :T] if T <= self.block_size else None
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits if loss is None else (logits, loss)
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, 
                 top_k: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor:
        """
        Generate text tokens autoregressively.
        
        Args:
            idx: Input token indices (B, T)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated token indices (B, T + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop sequence if it exceeds block size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Forward pass
            with torch.no_grad():
                logits = self(idx_cond)
                
                # Get logits for the last token and apply temperature
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter back to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample from the distribution
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


def create_model_from_config(config: dict) -> SimpleTransformer:
    """Create model from configuration dictionary."""
    model_config = config['model']
    
    return SimpleTransformer(
        vocab_size=model_config['vocab_size'],
        n_embd=model_config['embedding_dim'],
        n_layer=model_config['num_layers'],
        n_head=model_config['num_heads'],
        block_size=model_config['max_seq_length'],
        dropout=model_config.get('dropout', 0.1)
    )


if __name__ == "__main__":
    # Test the model
    import json
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Create model
    model = create_model_from_config(config)
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    vocab_size = config['model']['vocab_size']
    
    # Random input
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        logits = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {logits.shape}")
        print(f"Model parameters: {model.get_num_params():,}")
    
    # Test generation
    print("\nTesting generation...")
    generated = model.generate(x[:1, :5], max_new_tokens=10, temperature=0.8)
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")