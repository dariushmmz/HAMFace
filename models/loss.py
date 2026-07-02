"""loss.py — Hardness-Aware Margin (HAM) face loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import N_CLASSES, EMBED_DIM, LOSS_SCALE, LOSS_MARGIN, LOSS_HARDNESS


class HAMFaceLoss(nn.Module):
    """
    Hardness-Aware Margin Face Loss (HAMFace).

    Adapts the angular margin per sample based on how close the hardest
    negative class is, making the margin larger for easy samples and
    smaller for hard ones where over-penalisation would be harmful.

    Parameters
    ----------
    num_classes:
        Number of identity classes.
    s:
        Logit scale factor.
    m:
        Base angular margin (radians).
    t:
        Hardness coefficient — scales the adaptive margin component.
    embed_dim:
        Dimension of the L2-normalised embedding vectors.
    """

    def __init__(
        self,
        num_classes: int = N_CLASSES,
        s: float         = LOSS_SCALE,
        m: float         = LOSS_MARGIN,
        t: float         = LOSS_HARDNESS,
        embed_dim: int   = EMBED_DIM,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.s           = s
        self.m           = m
        self.t           = t
        self.embed_dim   = embed_dim
        self.W           = nn.Parameter(torch.randn(num_classes, embed_dim))

    def forward(self, embeddings: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Note: PyTorch convention is (predictions, targets), unlike Keras (targets, preds)
        y_true = y_true.long()

        embeddings = F.normalize(embeddings, p=2, dim=1)
        W_norm     = F.normalize(self.W,     p=2, dim=1)

        cos_theta = embeddings @ W_norm.T                        # (B, num_classes)
        theta     = torch.acos(cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7))

        batch_indices = torch.arange(y_true.size(0), device=y_true.device)
        theta_yi      = theta[batch_indices, y_true]             # (B,)

        # Hardest negative: closest non-target class
        one_hot         = F.one_hot(y_true, self.num_classes).float()
        masked_theta    = theta + 1e6 * one_hot
        min_inter_theta = masked_theta.min(dim=1).values        # (B,)

        # Adaptive margin
        hardness        = (theta_yi + self.m > min_inter_theta).float()
        s_x             = 1.0 - torch.cos(theta_yi)
        adaptive_margin = self.m + self.t * hardness * s_x

        # Modified target logit
        cos_theta_yi_mod = torch.cos(theta_yi + adaptive_margin)
        logits           = self.s * cos_theta.clone()
        logits[batch_indices, y_true] = cos_theta_yi_mod * self.s

        return F.cross_entropy(logits, y_true)