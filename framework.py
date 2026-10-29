"""
S2D FusionNet: Sequence-to-Two-Dimensional Feature Fusion Network.

Official PyTorch implementation of S2D FusionNet for radar High-Resolution
Range Profile (HRRP) target recognition.

Architecture:
    1. Dimension Converter (S2D Mapping & 2D Convolutional Downsampling)
    2. Dynamic KNN Graph Convolutional Network (GCN)
    3. Multi-Source Cross-Attention (MSCA) with Rotary Position Embedding (RoPE)
    4. Attention-based Global Pooling and Classification Head
"""

from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .dim_converter import DimensionConverter
    from .gcn import GCN
    from .multisource_transformer import MultiSourceTransformer
except (ImportError, ValueError):
    from dim_converter import DimensionConverter
    from gcn import GCN
    from multisource_transformer import MultiSourceTransformer


class S2DFusionNet(nn.Module):
    """
    S2D FusionNet Model for Radar HRRP Target Recognition.

    Args:
        method: 2D representation mapping strategy in ['RPM', 'GASF', 'GADF', 'Hybrid'].
                Default: 'RPM'.
        k: Number of nearest neighbors in dynamic graph construction. Default: 12.
        heads: Number of attention heads in multi-source Transformer. Default: 4.
        num_classes: Number of target categories. Default: 10.
        dim: Hidden feature dimension. Default: 128.
        gcn_layers: Number of stacked GCN layers. Default: 1.
        tf_layers: Number of stacked MultiSourceTransformer layers. Default: 1.
        petype: Positional encoding type in ['rope', 'tra', 'sin', 'none']. Default: 'rope'.
        dropout: Dropout probability. Default: 0.1.
    """

    def __init__(
        self,
        method: str = "RPM",
        k: int = 12,
        heads: int = 4,
        num_classes: int = 10,
        dim: int = 128,
        gcn_layers: int = 1,
        tf_layers: int = 1,
        petype: str = "rope",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.method = method
        self.k = k
        self.heads = heads
        self.num_classes = num_classes
        self.dim = dim
        self.gcn_layers_count = gcn_layers
        self.tf_layers_count = tf_layers
        self.petype = petype

        # 1. Dimension Converter module
        self.dc_block = DimensionConverter(method=self.method, dim=self.dim)

        # 2. Dynamic Graph Convolution blocks with LayerNorm
        self.gcn_blocks = nn.ModuleList([GCN(k=self.k, dim=self.dim) for _ in range(gcn_layers)])
        self.gcn_norms = nn.ModuleList([nn.LayerNorm(self.dim) for _ in range(gcn_layers)])

        # 3. Multi-Source Transformer blocks
        self.tf_blocks = nn.ModuleList(
            [
                MultiSourceTransformer(
                    heads=self.heads,
                    dim=self.dim,
                    petype=petype,
                    dropout=dropout,
                )
                for _ in range(tf_layers)
            ]
        )

        # 4. Attention-based Pooling and Linear Classification Head
        self.attention_pool = nn.Linear(self.dim, 1)
        self.class_head = nn.Linear(self.dim, self.num_classes)

    def forward(
        self, input_x: torch.Tensor, return_feature: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of S2D FusionNet.

        Args:
            input_x: 1D HRRP input sequence tensor of shape (Batch_size, 1, Length).
            return_feature: If True, also returns the pooled global representation
                            prior to the classifier head (useful for t-SNE / downstream tasks).

        Returns:
            If return_feature is False:
                logits: Classification logits of shape (Batch_size, num_classes).
            If return_feature is True:
                tuple of (logits, global_features) where global_features has shape (Batch_size, Dim).
        """
        # Step 1: Extract local tokens via DimensionConverter -> (B, Dim, N_tokens)
        local_tokens = self.dc_block(input_x)

        # Step 2: Dynamic graph relational modeling via GCN
        graph_tokens = local_tokens
        for gcn_layer, norm_layer in zip(self.gcn_blocks, self.gcn_norms):
            residual = graph_tokens
            # Norm along channel dimension: (B, Dim, N) -> (B, N, Dim) -> norm -> (B, Dim, N)
            normed = norm_layer(graph_tokens.transpose(-1, -2)).transpose(-1, -2)
            graph_tokens = residual + gcn_layer(normed)

        # Transpose to (B, N_tokens, Dim) for Transformer attention
        local_seq = local_tokens.transpose(-1, -2)
        graph_seq = graph_tokens.transpose(-1, -2)

        # Step 3: Multi-Source Cross-Attention (MSCA) fusion
        fused_tokens = local_seq
        for tf_layer in self.tf_blocks:
            fused_tokens = tf_layer(x_gra=graph_seq, x_ori=fused_tokens)

        # Step 4: Attention-weighted Global Pooling
        # Attention scores across tokens: (B, N_tokens, 1) -> (B, 1, N_tokens)
        att_scores = self.attention_pool(fused_tokens)
        att_weights = F.softmax(att_scores, dim=1).transpose(-1, -2)

        # Weighted sum: (B, 1, N_tokens) @ (B, N_tokens, Dim) -> (B, 1, Dim) -> (B, Dim)
        global_feature = torch.matmul(att_weights, fused_tokens).squeeze(1)

        # Classification logits
        logits = self.class_head(global_feature)

        if return_feature:
            return logits, global_feature
        return logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Create dummy HRRP sequence batch: 8 samples, 1 channel, 64 sequence length
    dummy_hrrp = torch.randn(8, 1, 64, device=device)

    model = S2DFusionNet(
        method="RPM",
        k=12,
        heads=4,
        num_classes=10,
        dim=128,
        gcn_layers=1,
        tf_layers=1,
        petype="rope",
    ).to(device)

    logits, features = model(dummy_hrrp, return_feature=True)
    print("Input shape:          ", dummy_hrrp.shape)
    print("Output logits shape:  ", logits.shape)
    print("Global features shape:", features.shape)
    print("Model parameter count:", sum(p.numel() for p in model.parameters() if p.requires_grad))
