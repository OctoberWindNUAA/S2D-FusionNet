"""
Positional encoding modules for sequence and token representations.

This module provides:
- TrainableEncoding: Learnable absolute positional embeddings.
- SinusoidalEncoding: Fixed sinusoidal frequency-based positional embeddings.
- RotaryEmbedding: Rotary Position Embedding (RoPE) for relative position awareness.
"""

import torch
import torch.nn as nn


class TrainableEncoding(nn.Module):
    """
    Learnable absolute positional encoding.

    Args:
        dim: Dimensionality of feature tokens.
        max_pos: Maximum sequence length (number of tokens). Default: 16.
    """

    def __init__(self, dim: int = 128, max_pos: int = 16):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_pos, dim))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learnable positional embeddings to input sequence.

        Args:
            x: Input tensor of shape (Batch_size, Seq_len, Dim).

        Returns:
            Position-augmented tensor of shape (Batch_size, Seq_len, Dim).
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class SinusoidalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding.

    Args:
        max_pos: Maximum sequence length. Default: 16.
        embed_dim: Dimensionality of feature tokens. Default: 128.
    """

    def __init__(self, max_pos: int = 16, embed_dim: int = 128):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        pos = torch.arange(max_pos).float().unsqueeze(1)
        pe_matrix = pos * inv_freq
        pe = torch.zeros(max_pos, embed_dim)
        pe[:, 0::2] = torch.sin(pe_matrix)
        pe[:, 1::2] = torch.cos(pe_matrix)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add sinusoidal positional embeddings to input sequence.

        Args:
            x: Input tensor of shape (Batch_size, Seq_len, Dim).

        Returns:
            Position-augmented tensor of shape (Batch_size, Seq_len, Dim).
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Applies multiplicative rotary matrices to Query and Key representations
    in attention mechanisms to inject relative positional information.

    Args:
        max_pos: Maximum sequence length. Default: 16.
        embed_dim: Dimensionality per attention head (feature_dim // heads). Default: 32.
    """

    def __init__(self, max_pos: int = 16, embed_dim: int = 32):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        pos = torch.arange(max_pos).float()
        freqs = torch.outer(pos, inv_freq)
        emb = torch.repeat_interleave(freqs, 2, dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dimensions of the input."""
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE transformation to input Query or Key tensor.

        Args:
            x: Tensor of shape (Batch_size, Heads, Seq_len, Head_dim).

        Returns:
            Rotated tensor of shape (Batch_size, Heads, Seq_len, Head_dim).
        """
        seq_len = x.size(2)
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        return (x * cos) + (self.rotate_half(x) * sin)


if __name__ == "__main__":
    dummy = torch.randn(2, 16, 128)
    tra = TrainableEncoding(dim=128, max_pos=16)
    sin = SinusoidalEncoding(max_pos=16, embed_dim=128)
    rope = RotaryEmbedding(max_pos=16, embed_dim=32)

    print("Trainable output shape:", tra(dummy).shape)
    print("Sinusoidal output shape:", sin(dummy).shape)

    dummy_head = torch.randn(2, 4, 16, 32)
    print("RoPE output shape:", rope(dummy_head).shape)
