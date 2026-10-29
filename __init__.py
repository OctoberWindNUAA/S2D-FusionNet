"""
S2D FusionNet: Sequence-to-Two-Dimensional Feature Fusion Network for Radar HRRP Target Recognition.

Official open-source PyTorch implementation.
"""

from .framework import S2DFusionNet
from .dim_converter import DimensionConverter
from .gcn import GCN
from .multisource_transformer import MultiSourceTransformer, MultiSourceAttention
from .position_encoding import RotaryEmbedding, TrainableEncoding, SinusoidalEncoding
from .utils import rpm, gasf, gadf, hybrid

__version__ = "1.0.0"
__author__ = "S2DFusionNet Authors"

__all__ = [
    "S2DFusionNet",
    "DimensionConverter",
    "GCN",
    "MultiSourceTransformer",
    "MultiSourceAttention",
    "RotaryEmbedding",
    "TrainableEncoding",
    "SinusoidalEncoding",
    "rpm",
    "gasf",
    "gadf",
    "hybrid",
]
