"""attention.py — channel attention, spatial attention, and dynamic fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CA_REDUCTION_RATIO


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention with residual connection."""

    def __init__(self, channels: int, reduction_ratio: int = CA_REDUCTION_RATIO):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg = x.mean(dim=(-2, -1))                  # (B, C)
        mx  = x.amax(dim=(-2, -1))                  # (B, C)
        attention = self.mlp(avg + mx)               # (B, C)
        attention = attention.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * attention + x                     # residual


class SpatialAttention(nn.Module):
    """Spatial attention via channel-wise pooling with residual connection."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg_pool = x.mean(dim=1, keepdim=True)       # (B, 1, H, W)
        max_pool = x.amax(dim=1, keepdim=True)       # (B, 1, H, W)
        combined = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, H, W)
        attention = torch.sigmoid(self.conv(combined))     # (B, 1, H, W)
        return x * attention + x                     # residual


class DynamicAttentionFusion(nn.Module):
    """
    Learn per-sample weights to fuse two feature maps of equal shape.

    The gating network sees the concatenation of both inputs and outputs
    a 2-element softmax, then blends the inputs accordingly.
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.dense = nn.Linear(in_features * 2, 2)

    def forward(
        self, channel_features: torch.Tensor, spatial_features: torch.Tensor
    ) -> torch.Tensor:
        concat  = torch.cat([channel_features, spatial_features], dim=-1)
        weights = F.softmax(self.dense(concat), dim=-1)   # (B, 2)
        w_ch    = weights[:, :1]                          # (B, 1)
        w_sp    = weights[:, 1:]                          # (B, 1)
        return w_ch * channel_features + w_sp * spatial_features