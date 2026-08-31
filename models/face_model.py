"""face_model.py — full HAMFace model construction and weight loading."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from facenet_pytorch import InceptionResnetV1
except ImportError as e:
    raise ImportError(
        "LocalStream now uses InceptionResnetV1 (VGGFace2-pretrained) as its "
        "backbone instead of EfficientNetB0. Install it with:\n"
        "    pip install facenet-pytorch\n"
    ) from e

from config import IMAGE_SIZE, N_CLASSES, EMBED_DIM, MODEL_WEIGHTS_PATH, CLASS_WEIGHTS_PATH
from .attention import ChannelAttention, SpatialAttention, DynamicAttentionFusion
from .cvt import CvT
from .loss import HAMFaceLoss

# Bottleneck width used for the per-stream projections before the final
# fusion step. Previously this was 64, which — combined with the wide
# local stream and the global stream output — collapsed most of the
# discriminative signal before DynamicAttentionFusion ever saw it.
PROJ_DIM = 256

# InceptionResnetV1's last two blocks before the embedding head — the
# ones most likely to hold identity-discriminative detail worth
# fine-tuning on your dataset, analogous to the last MBConv stages we
# used to unfreeze on EfficientNetB0.
DEFAULT_UNFROZEN_BLOCKS = ("repeat_3", "block8")


def _build_cvt(n_classes: int) -> CvT:
    """
    Construct the CvT global stream, tolerant of either the classic
    signature (``CvT(in_channels, num_classes)``, softmax classification
    head) or the updated one (``CvT(in_channels)``, raw pooled features,
    no head). Avoids this file silently going stale if cvt.py's
    constructor signature changes.
    """
    try:
        return CvT(in_channels=3, num_classes=n_classes)
    except TypeError:
        return CvT(in_channels=3)


def _infer_cvt_output_dim(cvt: CvT, image_size: int) -> int:
    """
    Run a single dummy forward pass to determine the CvT's actual output
    feature dimension, rather than assuming it equals n_classes. Keeps
    face_model.py correct whether cvt.py still has an internal
    classification head or returns pooled pre-head features.
    """
    was_training = cvt.training
    cvt.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size)
        out = cvt(dummy, False)
    cvt.train(was_training)
    return out.shape[-1]


class L2Normalization(nn.Module):
    """Divide each embedding vector by its L2 norm along the last axis."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)


class LocalStream(nn.Module):
    """
    Face-pretrained backbone (InceptionResnetV1 / VGGFace2) with channel
    + spatial attention.

    Why this backbone instead of EfficientNetB0
    ---------------------------------------------
    EfficientNetB0's ImageNet weights encode general object features
    (edges, textures, object parts for cats/cars/etc.), not features
    tuned to discriminate between human faces. InceptionResnetV1 here is
    loaded with weights pretrained on VGGFace2 (3.3M face images across
    9131 identities) — it already "knows" what makes two faces different,
    which is a much better starting point than ImageNet features finetuned
    on a comparatively small identity set.

    We deliberately skip the model's own embedding head (avgpool ->
    last_linear -> last_bn -> logits) and instead pull the spatial feature
    map straight out of ``block8`` (shape ``(B, 1792, H', W')``), then run
    our own channel/spatial attention + fusion on top of it, same as
    before. The backbone starts fully frozen; call
    ``unfreeze_last_blocks()`` after a warmup phase to fine-tune the last
    two blocks (``repeat_3``, ``block8``) on your face data.
    """

    # Channel count of InceptionResnetV1's block8 output (pre-pool feature map)
    BACKBONE_CHANNELS = 1792

    # Named children of InceptionResnetV1, in forward order, up to (and
    # including) block8 — i.e. everything except its own pooling/embedding
    # head (avgpool_1a, dropout, last_linear, last_bn, logits), which we
    # don't use.
    _FORWARD_BLOCKS = (
        "conv2d_1a", "conv2d_2a", "conv2d_2b", "maxpool_3a",
        "conv2d_3b", "conv2d_4a", "conv2d_4b",
        "repeat_1", "mixed_6a", "repeat_2", "mixed_7a", "repeat_3", "block8",
    )

    def __init__(self, pretrained: str | None = "vggface2"):
        super().__init__()
        if pretrained is None:
            # Match the module layout of the VGGFace2 model without loading its
            # separate initialization checkpoint. The HAMFace state dict owns
            # all of these parameters, including the unused logits module.
            self.backbone = InceptionResnetV1(
                pretrained=None,
                classify=True,
                num_classes=8631,
            )
            self.backbone.classify = False
        else:
            self.backbone = InceptionResnetV1(pretrained=pretrained)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._unfrozen_blocks: tuple[str, ...] = ()

        C = self.BACKBONE_CHANNELS
        self.channel_attention = ChannelAttention(channels=C)
        self.spatial_attention = SpatialAttention()
        self.fusion            = DynamicAttentionFusion(in_features=C)

    def _extract_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Run input through InceptionResnetV1 up to block8, skipping its embedding head."""
        for name in self._FORWARD_BLOCKS:
            x = getattr(self.backbone, name)(x)
        return x  # (B, 1792, H', W')

    def unfreeze_last_blocks(self, block_names: tuple[str, ...] = DEFAULT_UNFROZEN_BLOCKS) -> None:
        """
        Unfreeze the named top-level blocks of the backbone for
        fine-tuning. Intended to be called after a warmup phase (e.g. a
        few epochs training only the attention/fusion/embedding heads),
        so the rest of the network has stabilized before backbone
        gradients start flowing.

        Use a lower learning rate for these params than for the rest of
        the model (e.g. 10x smaller) — see ``HAMFace.get_param_groups``.
        """
        self._unfrozen_blocks = tuple(block_names)
        for name, module in self.backbone.named_children():
            if name in self._unfrozen_blocks:
                for p in module.parameters():
                    p.requires_grad = True

    def backbone_trainable_parameters(self):
        """Yield only the currently-unfrozen backbone parameters (for param groups)."""
        for name, module in self.backbone.named_children():
            if name in self._unfrozen_blocks:
                yield from module.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)  → features: (B, C, h, w)
        feat    = self._extract_feature_map(x)
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
    * **Local stream** — InceptionResnetV1 (VGGFace2-pretrained; last two
      blocks unfreezable for fine-tuning) with channel- and spatial-
      attention, fused by ``DynamicAttentionFusion``.
    * **Global stream** — single-stage CvT. Its output feature dimension
      is auto-detected via a dummy forward pass rather than assumed, so
      this file stays correct whether cvt.py still has an internal
      classification head or returns raw pooled features.
    * Both streams are projected to ``PROJ_DIM``-d, fused again, then
      mapped to an ``EMBED_DIM``-d L2-normalised embedding.

    Parameters
    ----------
    n_classes:
        Number of identity classes (passed through to CvT for backward
        compatibility with the classic head-based signature, if present).
    """

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        image_size: int = IMAGE_SIZE,
        local_pretrained: str | None = "vggface2",
    ):
        super().__init__()
        self.local_stream = LocalStream(pretrained=local_pretrained)
        self.cvt           = _build_cvt(n_classes)

        local_dim  = LocalStream.BACKBONE_CHANNELS
        global_dim = _infer_cvt_output_dim(self.cvt, image_size)

        self.local_proj    = nn.Sequential(nn.Linear(local_dim,  PROJ_DIM), nn.ReLU())
        self.global_proj   = nn.Sequential(nn.Linear(global_dim, PROJ_DIM), nn.ReLU())
        self.final_fusion  = DynamicAttentionFusion(in_features=PROJ_DIM)
        self.embedding      = nn.Linear(PROJ_DIM, EMBED_DIM)
        self.l2_norm        = L2Normalization()

    def unfreeze_backbone(self, block_names: tuple[str, ...] = DEFAULT_UNFROZEN_BLOCKS) -> None:
        """Convenience passthrough to unfreeze the local stream's last backbone blocks."""
        self.local_stream.unfreeze_last_blocks(block_names)

    def get_param_groups(self, base_lr: float, backbone_lr_mult: float = 0.1) -> list[dict]:
        """
        Build optimizer param groups with a reduced LR for any unfrozen
        backbone parameters. Call this AFTER ``unfreeze_backbone()`` so the
        backbone group is non-empty when you actually want to fine-tune it.

        Example
        -------
        >>> model.unfreeze_backbone()
        >>> optimizer = torch.optim.AdamW(model.get_param_groups(base_lr=3e-4))
        """
        backbone_params = list(self.local_stream.backbone_trainable_parameters())
        backbone_ids    = {id(p) for p in backbone_params}
        rest_params     = [p for p in self.parameters() if p.requires_grad and id(p) not in backbone_ids]

        groups = [{"params": rest_params, "lr": base_lr}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": base_lr * backbone_lr_mult})
        return groups

    def forward(
        self,
        local_input: torch.Tensor,
        cvt_input: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
        # local_input, cvt_input: (B, 3, H, W)  [channels-first for PyTorch]
        local_feat  = self.local_proj(self.local_stream(local_input))   # (B, PROJ_DIM)
        global_feat = self.global_proj(self.cvt(cvt_input, training))   # (B, PROJ_DIM)
        combined    = self.final_fusion(local_feat, global_feat)        # (B, PROJ_DIM)
        embedding   = self.l2_norm(self.embedding(combined))           # (B, EMBED_DIM)
        return embedding


def build_model(
    n_classes: int = N_CLASSES,
    local_pretrained: str | None = "vggface2",
) -> HAMFace:
    """Construct and return the HAMFace model."""
    return HAMFace(n_classes=n_classes, local_pretrained=local_pretrained)


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

    # The saved state dict includes the complete local backbone. Avoid fetching
    # the separate VGGFace2 initialization weights during normal inference.
    model = build_model(n_classes, local_pretrained=None)
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
