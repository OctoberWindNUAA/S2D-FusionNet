"""
Mathematical transformation utilities for Sequence-to-Two-Dimensional (S2D) conversion.

This module provides tensor implementations of:
- Relative Position Matrix (RPM)
- Gramian Angular Summation Field (GASF)
- Gramian Angular Difference Field (GADF)
- Hybrid multi-channel representation combining RPM, GASF, and GADF.
"""

from typing import Union
import torch


def rpm(input_x: torch.Tensor) -> torch.Tensor:
    """
    Compute Relative Position Matrix (RPM) for 1D sequences.

    For a sequence x of length L, RPM models pairwise relative amplitude differences:
        RPM[b, c, i, j] = x[b, c, i] - x[b, c, j]

    Args:
        input_x: Input tensor of shape (Batch_size, Channels, Length).

    Returns:
        Tensor of shape (Batch_size, Channels, Length, Length).
    """
    return input_x.unsqueeze(2) - input_x.unsqueeze(3)


def gasf(input_x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Gramian Angular Summation Field (GASF) for 1D sequences.

    Encodes temporal sequences into polar coordinates and computes angular summation:
        GASF[b, c, i, j] = cos(theta_i + theta_j)

    Args:
        input_x: Input tensor of shape (Batch_size, Channels, Length).
        eps: Small epsilon to prevent division by zero during normalization.

    Returns:
        Tensor of shape (Batch_size, Channels, Length, Length).
    """
    min_val = input_x.min(dim=-1, keepdim=True)[0]
    max_val = input_x.max(dim=-1, keepdim=True)[0]
    input_norm = 2.0 * (input_x - min_val) / (max_val - min_val + eps) - 1.0
    input_norm = torch.clamp(input_norm, -1.0, 1.0)
    theta = torch.arccos(input_norm)
    theta_i = theta.unsqueeze(3)
    theta_j = theta.unsqueeze(2)
    return torch.cos(theta_i + theta_j)


def gadf(input_x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Gramian Angular Difference Field (GADF) for 1D sequences.

    Encodes temporal sequences into polar coordinates and computes angular difference:
        GADF[b, c, i, j] = sin(theta_i - theta_j)

    Args:
        input_x: Input tensor of shape (Batch_size, Channels, Length).
        eps: Small epsilon to prevent division by zero during normalization.

    Returns:
        Tensor of shape (Batch_size, Channels, Length, Length).
    """
    min_val = input_x.min(dim=-1, keepdim=True)[0]
    max_val = input_x.max(dim=-1, keepdim=True)[0]
    input_norm = 2.0 * (input_x - min_val) / (max_val - min_val + eps) - 1.0
    input_norm = torch.clamp(input_norm, -1.0, 1.0)
    theta = torch.arccos(input_norm)
    theta_i = theta.unsqueeze(3)
    theta_j = theta.unsqueeze(2)
    return torch.sin(theta_i - theta_j)


def hybrid(input_x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Hybrid 3-channel 2D representation combining RPM, GASF, and GADF.

    Args:
        input_x: Input tensor of shape (Batch_size, Channels, Length).
        eps: Small epsilon for numerical stability.

    Returns:
        Concatenated tensor of shape (Batch_size, 3 * Channels, Length, Length).
    """
    m_rpm = rpm(input_x)
    m_gasf = gasf(input_x, eps=eps)
    m_gadf = gadf(input_x, eps=eps)
    return torch.cat([m_rpm, m_gasf, m_gadf], dim=1)


if __name__ == "__main__":
    dummy_input = torch.randn(4, 1, 64)
    print("Input shape:", dummy_input.shape)
    print("RPM shape:   ", rpm(dummy_input).shape)
    print("GASF shape:  ", gasf(dummy_input).shape)
    print("GADF shape:  ", gadf(dummy_input).shape)
    print("Hybrid shape:", hybrid(dummy_input).shape)
