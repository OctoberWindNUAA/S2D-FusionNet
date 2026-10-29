"""
Multi-Source Cross-Attention (MSCA) and Transformer interaction modules.

This module implements:
- MultiSourceAttention: Asymmetric cross-attention mechanism aligning local
  scattering tokens (Query) with graph-enhanced topological tokens (Key & Value).
- MultiSourceTransformer: Complete Transformer block with LayerNorm, MSCA,
  optional positional encodings (RoPE / Trainable / Sinusoidal), and FFN.
"""

import math
from typing import Optional
import torch
import torch.nn as nn

try:
    from .position_encoding import TrainableEncoding, SinusoidalEncoding, RotaryEmbedding
except (ImportError, ValueError):
    from position_encoding import TrainableEncoding, SinusoidalEncoding, RotaryEmbedding


class MultiSourceAttention(nn.Module):
    """
    Multi-Source Cross-Attention (MSCA) module.

    Uses the local token sequence as Query and the graph-enhanced sequence as Key and Value:
        Attention(Q_local, K_graph, V_graph) = softmax(Q_local @ K_graph^T / sqrt(d_k)) @ V_graph

    Args:
        feature_dim: Channel dimension of the tokens.
        heads: Number of attention heads.
        dropout: Attention dropout rate. Default: 0.1.
    """

    def __init__(self, feature_dim: int = 128, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert feature_dim % heads == 0, f"feature_dim ({feature_dim}) must be divisible by heads ({heads})"
        self.heads = heads
        self.dim = feature_dim
        self.head_dim = feature_dim // heads

        self.q = nn.Linear(feature_dim, feature_dim)
        self.k = nn.Linear(feature_dim, feature_dim)
        self.v = nn.Linear(feature_dim, feature_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(feature_dim, feature_dim)
        self.out_drop = nn.Dropout(dropout)

    def forward(
        self,
        x_ori: torch.Tensor,
        x_gra: torch.Tensor,
        rope_module: Optional[nn.Module] = None,
        use_rope: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of MultiSourceAttention.

        Args:
            x_ori: Local feature source (Query) of shape (Batch_size, Seq_len, Dim).
            x_gra: Graph-enhanced feature source (Key, Value) of shape (Batch_size, Seq_len, Dim).
            rope_module: Optional RotaryEmbedding module.
            use_rope: Whether to apply rotary position embedding to Q and K.

        Returns:
            Fused attention representation of shape (Batch_size, Seq_len, Dim).
        """
        b, n, c = x_ori.shape
        q = self.q(x_ori).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x_gra).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x_gra).view(b, n, self.heads, self.head_dim).transpose(1, 2)

        if use_rope and rope_module is not None:
            q = rope_module(q)
            k = rope_module(k)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(b, n, c)
        return self.out_drop(self.proj(context))


class MultiSourceTransformer(nn.Module):
    """
    Multi-Source Transformer Block.

    Integrates local scattering features and graph-enhanced topological features
    through multi-source cross-attention, followed by a feed-forward network.

    Args:
        heads: Number of attention heads. Default: 4.
        dim: Token feature dimensionality. Default: 128.
        petype: Positional encoding type in ['rope', 'tra', 'sin', 'none']. Default: 'rope'.
        dropout: Dropout rate. Default: 0.1.
    """

    def __init__(
        self,
        heads: int = 4,
        dim: int = 128,
        petype: str = "rope",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.petype = petype.lower()
        self.dim = dim
        self.heads = heads

        # Positional encoding options
        self.tra_gra = TrainableEncoding(dim=dim) if self.petype == "tra" else None
        self.tra_ori = TrainableEncoding(dim=dim) if self.petype == "tra" else None
        self.sin_enc = SinusoidalEncoding(max_pos=16, embed_dim=dim) if self.petype == "sin" else None
        self.rope_enc = (
            RotaryEmbedding(max_pos=16, embed_dim=dim // heads) if self.petype == "rope" else None
        )

        self.attn = MultiSourceAttention(feature_dim=dim, heads=heads, dropout=dropout)
        self.norm_gra = nn.LayerNorm(dim, eps=1e-6)
        self.norm_ori = nn.LayerNorm(dim, eps=1e-6)
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)

        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x_gra: torch.Tensor, x_ori: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MultiSourceTransformer.

        Args:
            x_gra: Graph-enhanced tokens of shape (Batch_size, Seq_len, Dim).
            x_ori: Local scattering tokens of shape (Batch_size, Seq_len, Dim).

        Returns:
            Fused output tokens of shape (Batch_size, Seq_len, Dim).
        """
        x_gra_norm = self.norm_gra(x_gra)
        x_ori_norm = self.norm_ori(x_ori)

        if self.petype == "tra":
            x_gra_norm = self.tra_gra(x_gra_norm)
            x_ori_norm = self.tra_ori(x_ori_norm)
        elif self.petype == "sin":
            x_gra_norm = self.sin_enc(x_gra_norm)
            x_ori_norm = self.sin_enc(x_ori_norm)

        use_rope = (self.petype == "rope")
        att_out = self.attn(
            x_ori=x_ori_norm,
            x_gra=x_gra_norm,
            rope_module=self.rope_enc,
            use_rope=use_rope,
        )

        # Residual connection with local tokens
        x = x_ori + att_out

        # Feed-forward network with residual connection
        x = x + self.ffn(self.norm_ffn(x))
        return x


if __name__ == "__main__":
    dummy_ori = torch.randn(4, 16, 128)
    dummy_gra = torch.randn(4, 16, 128)
    transformer = MultiSourceTransformer(heads=4, dim=128, petype="rope")
    out = transformer(x_gra=dummy_gra, x_ori=dummy_ori)
    print("MultiSourceTransformer Output shape:", out.shape)
