"""cvt.py — Convolutional Vision Transformer (CvT) building blocks.

Memory budget for self-attention
---------------------------------
Attention matrix shape: (batch, heads, seq_len, seq_len)
With IMAGE_SIZE=128 and a single 2x downsample: seq_len = 64*64 = 4096
  → 8 * 1 * 4096 * 4096 * 4 bytes ≈ 536 MB per batch  →  OOM on CPU

Fix: add a second ConvEmbedding (stride=2) before the transformer so the
spatial map is 16x16 = 256 tokens before attention is computed.
  → 8 * 1 * 256 * 256 * 4 bytes ≈ 2 MB  ✓
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    IMAGE_SIZE,
    CVT_EMBED_DIM,
    CVT_NUM_HEADS,
    CVT_FF_DIM,
    CVT_DROPOUT,
)


class ConvEmbedding(nn.Module):
    """Conv projection into *embed_dim* channels with an optional 2× spatial downsample."""

    def __init__(self, in_channels: int, embed_dim: int, downsample: bool = True):
        super().__init__()
        self.embed_dim  = embed_dim
        self.downsample = downsample
        stride = 2 if downsample else 1
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.conv(x)                             # (B, embed_dim, H', W')
        # LayerNorm expects (B, *, C): permute → norm → permute back
        x = x.permute(0, 2, 3, 1)                   # (B, H', W', embed_dim)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)                # (B, embed_dim, H', W')


class TransformerEncoder(nn.Module):
    """
    Transformer encoder block operating on spatial feature maps.

    Flattens (B, C, H, W) → (B, H*W, C) for attention, then reshapes back.
    Keep H*W small (≤ 256) to avoid OOM on CPU.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        rate: float = CVT_DROPOUT,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim    = ff_dim
        self.rate      = rate

        self.att        = nn.MultiheadAttention(embed_dim, num_heads, dropout=rate, batch_first=True)
        self.ffn        = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1   = nn.Dropout(rate)
        self.dropout2   = nn.Dropout(rate)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        # x: (B, C, H, W)
        self.att.training  = training
        self.ffn.training  = training

        B, C, H, W = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, seq_len, C)

        attn_out, _ = self.att(seq, seq, seq)
        attn_out    = self.dropout1(attn_out)
        out1        = self.layernorm1(seq + attn_out)

        ffn_out = self.dropout2(self.ffn(out1))
        out2    = self.layernorm2(out1 + ffn_out)

        return out2.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)


class CvT(nn.Module):
    """
    Two-stage CvT that keeps the attention sequence length manageable.

    Stage 1: 128×128 → 64×64  (stride-2 ConvEmbedding, no attention)
    Stage 2: 64×64  → 16×16   (stride-4 ConvEmbedding via two stride-2 convs,
                                 then TransformerEncoder on 16*16=256 tokens)

    Attention cost: batch * heads * 256 * 256 * 4 bytes ≈ 2 MB  (vs 2 GB before)
    """

    def __init__(
        self,
        in_channels: int  = 3,
        num_classes: int  = CVT_EMBED_DIM,
        image_size: int   = IMAGE_SIZE,
    ):
        super().__init__()
        self.embed_stage1 = ConvEmbedding(in_channels,          CVT_EMBED_DIM,     downsample=True)
        self.embed_stage2 = ConvEmbedding(CVT_EMBED_DIM,        CVT_EMBED_DIM * 2, downsample=True)
        self.embed_stage3 = ConvEmbedding(CVT_EMBED_DIM * 2,    CVT_EMBED_DIM * 2, downsample=True)
        self.transformer  = TransformerEncoder(
            embed_dim=CVT_EMBED_DIM * 2,
            num_heads=CVT_NUM_HEADS,
            ff_dim=CVT_FF_DIM * 2,
        )
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.head    = nn.Linear(CVT_EMBED_DIM * 2, num_classes)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        # x: (B, 3, H, W)
        x = self.embed_stage1(x)
        x = self.embed_stage2(x)
        x = self.embed_stage3(x)
        x = self.transformer(x, training=training)
        x = self.pool(x).flatten(1)   # (B, embed_dim*2)
        x = self.dropout(x)
        return F.softmax(self.head(x), dim=-1)