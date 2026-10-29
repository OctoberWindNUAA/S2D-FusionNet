"""
Dynamic Graph Convolutional Network (GCN) module.

This module implements a dynamic KNN graph convolution block (EdgeConv) that
infers non-Euclidean topological relationships among scattering tokens directly
in the learned feature space.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    """
    Basic 2D convolution block consisting of Conv2D (1x1), BatchNorm, and GELU.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GCN(nn.Module):
    """
    Dynamic K-Nearest Neighbor (KNN) Graph Convolution Block.

    For each node i and its dynamically identified k nearest neighbors j in feature space,
    the block constructs edge features [x_i, x_j - x_i], processes them with EdgeConv,
    aggregates over neighbors via max-pooling, and applies a residual connection:
        x_out = Linear(MaxPool_j(EdgeConv([x_i, x_j - x_i]))) + x_in

    Args:
        k: Number of nearest neighbors for dynamic graph construction. Default: 12.
        dim: Dimensionality of node features. Default: 128.
    """

    def __init__(self, k: int = 12, dim: int = 128):
        super().__init__()
        self.k = k
        self.dim = dim
        self.edge_conv = BasicConv(2 * dim, dim)
        self.linear = nn.Linear(dim, dim)

    @staticmethod
    def batched_index_select(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Gather features for specified indices across batches.

        Args:
            x: Tensor of shape (Batch_size, Dim, N_nodes).
            idx: Index tensor of shape (Batch_size, N_nodes, K).

        Returns:
            Gathered tensor of shape (Batch_size, Dim, N_nodes, K).
        """
        b, c, n = x.shape
        k = idx.shape[-1]
        idx_expanded = idx.unsqueeze(1).expand(-1, c, -1, -1)
        x_expanded = x.unsqueeze(-1).expand(-1, -1, -1, k)
        return torch.gather(x_expanded, dim=2, index=idx_expanded)

    def compute_knn(self, input_x_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute dynamic k-nearest neighbors based on squared Euclidean distances.

        Args:
            input_x_norm: L2-normalized node features of shape (Batch_size, N_nodes, Dim).

        Returns:
            Tuple of (neighbor_indices, center_indices), each of shape (Batch_size, N_nodes, K).
        """
        b, n, _ = input_x_norm.shape
        # ||x_i - x_j||^2 = ||x_i||^2 - 2 * <x_i, x_j> + ||x_j||^2
        x_inner = -2.0 * torch.matmul(input_x_norm, input_x_norm.transpose(2, 1))
        x_square = input_x_norm.pow(2).sum(dim=-1, keepdim=True)
        dist = x_square + x_inner + x_square.transpose(2, 1)

        # Retrieve top-k nearest neighbors (smallest distance corresponds to largest -dist)
        _, nn_idx = torch.topk(-dist, k=self.k, dim=-1)
        center_idx = torch.arange(0, n, device=input_x_norm.device).view(1, n, 1).expand(b, n, self.k)
        return nn_idx, center_idx

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of dynamic graph convolution.

        Args:
            input_x: Input node features of shape (Batch_size, Dim, N_nodes).

        Returns:
            Graph-enhanced node features of shape (Batch_size, Dim, N_nodes).
        """
        res = input_x
        # Normalize features along channel dimension prior to distance calculation
        input_x_norm = F.normalize(input_x, p=2.0, dim=1)

        with torch.no_grad():
            nn_idx, center_idx = self.compute_knn(input_x_norm.transpose(2, 1))

        # Gather center and neighbor features
        x_center = self.batched_index_select(input_x, center_idx)
        x_neighbor = self.batched_index_select(input_x, nn_idx)

        # Construct edge features: [x_i, x_j - x_i]
        edge_feature = torch.cat([x_center, x_neighbor - x_center], dim=1)

        # EdgeConv and max-pooling aggregation
        conv_out = self.edge_conv(edge_feature)
        aggregated, _ = torch.max(conv_out, dim=-1, keepdim=False)

        # Feature projection and residual connection
        projected = self.linear(aggregated.transpose(1, 2)).transpose(1, 2)
        out = projected + res
        return out


if __name__ == "__main__":
    dummy_tokens = torch.randn(4, 128, 16)
    gcn_block = GCN(k=12, dim=128)
    out_tokens = gcn_block(dummy_tokens)
    print("GCN Input shape: ", dummy_tokens.shape)
    print("GCN Output shape:", out_tokens.shape)
