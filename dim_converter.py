"""
Sequence-to-Two-Dimensional (S2D) Dimension Converter Module.

This module maps 1D temporal/spatial sequence signals into structured 2D matrix
representations (via RPM, GASF, GADF, or Hybrid) and applies a lightweight 2D
convolutional downsampling network to extract compact local scattering tokens.
"""

from typing import Literal
import torch
import torch.nn as nn

try:
    from .utils import rpm, gasf, gadf, hybrid
except (ImportError, ValueError):
    from utils import rpm, gasf, gadf, hybrid


class DimensionConverter(nn.Module):
    """
    Converts 1D sequences into 2D representations and extracts compact tokens.

    Architecture pipeline:
        1D Sequence (B, 1, L)
           -> 2D Matrix (B, C_in, L, L) via RPM / GASF / GADF / Hybrid
           -> Conv2D (k=7, s=2, p=3) -> ReLU -> MaxPool2D (k=3, s=2, p=1)
           -> Conv2D (k=7, s=2, p=3) -> ReLU -> MaxPool2D (k=3, s=2, p=1)
           -> Flatten spatial grid into sequence: (B, Dim, N_tokens).

    Args:
        method: 2D conversion strategy, one of ['RPM', 'GASF', 'GADF', 'Hybrid'].
        dim: Channel dimensionality of the output tokens. Default: 128.
    """

    def __init__(self, method: str = "RPM", dim: int = 128):
        super().__init__()
        self.method = method.upper()
        in_channels = 3 if self.method == "HYBRID" else 1

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DimensionConverter.

        Args:
            input_x: Input tensor of shape (Batch_size, 1, Length).

        Returns:
            Extracted token sequence of shape (Batch_size, Dim, N_tokens).
            For an input sequence length of 64, N_tokens = 16 (4x4 spatial grid).
        """
        if self.method == "RPM":
            x_2d = rpm(input_x)
        elif self.method == "GASF":
            x_2d = gasf(input_x)
        elif self.method == "GADF":
            x_2d = gadf(input_x)
        elif self.method in ["HYBRID", "HYB"]:
            x_2d = hybrid(input_x)
        else:
            raise ValueError(f"Unsupported 2D conversion method: {self.method}")

        features_2d = self.feature_extractor(x_2d)
        # Flatten (B, Dim, H, W) -> (B, Dim, H*W)
        tokens = torch.flatten(features_2d, start_dim=2)
        return tokens


if __name__ == "__main__":
    dummy_input = torch.randn(8, 1, 64)
    for m in ["RPM", "GASF", "GADF", "Hybrid"]:
        conv = DimensionConverter(method=m, dim=128)
        out = conv(dummy_input)
        print(f"Method: {m:<7} | Input: {tuple(dummy_input.shape)} -> Output: {tuple(out.shape)}")
