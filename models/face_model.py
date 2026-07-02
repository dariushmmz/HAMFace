"""face_model.py — full HAMFace model construction and weight loading."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from config import IMAGE_SIZE, N_CLASSES, EMBED_DIM, MODEL_WEIGHTS_PATH, CLASS_WEIGHTS_PATH
from .attention import ChannelAttention, SpatialAttention, DynamicAttentionFusion
from .cvt import CvT
from .loss import HAMFaceLoss


class L2Normalization(nn.Module):
    """Divide each embedding vector by its L2 norm along the last axis."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)


class LocalStream(nn.Module):
    """EfficientNetB0 backbone (frozen) with channel + spatial attention."""

    # EfficientNetB0 final feature map has 1280 channels (before classifier)
    BACKBONE_CHANNELS = 1280

    def __init__(self):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Drop classifier; keep only the feature extractor
        self.features = backbone.features
        self.avgpool  = backbone.avgpool  # AdaptiveAvgPool2d → not used directly here
        for p in self.features.parameters():
            p.requires_grad = False

        C = self.BACKBONE_CHANNELS
        self.channel_attention = ChannelAttention(channels=C)
        self.spatial_attention = SpatialAttention()
        self.fusion            = DynamicAttentionFusion(in_features=C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)  → features: (B, C, h, w)
        feat    = self.features(x)
        ch_feat = self.channel_attention(feat)          # (B, C, h, w)
        sp_feat = self.spatial_attention(feat)          # (B, C, h, w)

        # DynamicAttentionFusion expects flat vectors; pool first
        ch_flat = ch_feat.mean(dim=(-2, -1))            # (B, C)
        sp_flat = sp_feat.mean(dim=(-2, -1))            # (B, C)
        return self.fusion(ch_flat, sp_flat)            # (B, C)


class HAMFace(nn.Module):
    """
    Two-stream HAMFace model.

    Architecture
    ------------
    * **Local stream** — EfficientNetB0 (frozen ImageNet weights) with
      channel- and spatial-attention, fused by ``DynamicAttentionFusion``.
    * **Global stream** — single-stage CvT.
    * Both streams are projected to 64-d, fused again, then mapped to an
      ``EMBED_DIM``-d L2-normalised embedding.

    Parameters
    ----------
    n_classes:
        Number of identity classes (used by the CvT classification head).
    """

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.local_stream = LocalStream()
        self.cvt          = CvT(in_channels=3, num_classes=n_classes)

        local_dim  = LocalStream.BACKBONE_CHANNELS
        global_dim = n_classes  # CvT output size

        self.local_proj  = nn.Sequential(nn.Linear(local_dim,  64), nn.ReLU())
        self.global_proj = nn.Sequential(nn.Linear(global_dim, 64), nn.ReLU())
        self.final_fusion = DynamicAttentionFusion(in_features=64)
        self.embedding    = nn.Linear(64, EMBED_DIM)
        self.l2_norm      = L2Normalization()

    def forward(
        self,
        local_input: torch.Tensor,
        cvt_input: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
        # local_input, cvt_input: (B, 3, H, W)  [channels-first for PyTorch]
        local_feat  = self.local_proj(self.local_stream(local_input))   # (B, 64)
        global_feat = self.global_proj(self.cvt(cvt_input, training))   # (B, 64)
        combined    = self.final_fusion(local_feat, global_feat)        # (B, 64)
        embedding   = self.l2_norm(self.embedding(combined))           # (B, EMBED_DIM)
        return embedding


def build_model(n_classes: int = N_CLASSES) -> HAMFace:
    """Construct and return the HAMFace model."""
    return HAMFace(n_classes=n_classes)


def load_model(
    n_classes: int          = N_CLASSES,
    weights_path: str       = MODEL_WEIGHTS_PATH,
    class_weights_path: str = CLASS_WEIGHTS_PATH,
    device: str | None      = None,
) -> tuple[HAMFace, HAMFaceLoss]:
    """
    Build the model, restore saved weights, and return both the model and
    the HAMFaceLoss instance with its weight matrix loaded.

    Parameters
    ----------
    n_classes:
        Must match the number of classes used during training.
    weights_path:
        Path to the ``.pt`` / ``.pth`` checkpoint produced by ``train.py``.
    class_weights_path:
        Path to the ``.npy`` file containing the HAMFace weight matrix ``W``.
    device:
        Target device string (e.g. ``"cuda"``, ``"cpu"``).  Defaults to
        ``"cuda"`` when available, else ``"cpu"``.

    Returns
    -------
    (model, loss_fn)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(n_classes)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    print(f"[load_model] Weights loaded from '{weights_path}'.")

    loss_fn = HAMFaceLoss(num_classes=n_classes)
    class_weights = np.load(class_weights_path)
    with torch.no_grad():
        loss_fn.W.copy_(torch.from_numpy(class_weights))
    loss_fn.to(device)
    print(f"[load_model] HAMFaceLoss W loaded from '{class_weights_path}'.")

    return model, loss_fn