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

        batch_indices = torch.arange(y_true.size(0), device=y_true.device)
        cos_theta_yi  = cos_theta[batch_indices, y_true]         # (B,)

        # theta_yi is only needed for the hardness comparison and the s_x
        # weighting term below — acos is fine here since it isn't fed back
        # through cos() for the logit itself (see note further down).
        theta_yi = torch.acos(cos_theta_yi.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        theta    = torch.acos(cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7))

        # Hardest negative: closest non-target class
        one_hot         = F.one_hot(y_true, self.num_classes).float()
        masked_theta    = theta + 1e6 * one_hot
        min_inter_theta = masked_theta.min(dim=1).values          # (B,)

        # Adaptive margin
        # hardness == 1  ->  sample is HARD (target angle + base margin
        #                    already crosses into the nearest negative's
        #                    territory) -> must NOT receive extra margin.
        # hardness == 0  ->  sample is EASY -> safe to add extra margin to
        #                    keep the embedding space well structured.
        # (Previous version added the extra term when hardness == 1, i.e.
        # penalized hard samples the most — the opposite of the intended
        # behaviour. Fixed below by gating on (1 - hardness).)
        hardness        = (theta_yi + self.m > min_inter_theta).float()
        s_x             = 1.0 - torch.cos(theta_yi)
        adaptive_margin = self.m + self.t * (1.0 - hardness) * s_x

        # Modified target logit.
        # Compute cos(theta_yi + adaptive_margin) via the angle-sum identity
        # directly from cos_theta_yi (and its matching sin), instead of
        # round-tripping through acos -> cos. acos's gradient explodes near
        # +/-1 (d/dx acos(x) = -1/sqrt(1-x^2)), which is exactly where
        # confidently-correct samples sit late in training — this form
        # avoids that instability for the backward pass through the logit.
        sin_theta_yi = torch.sqrt((1.0 - cos_theta_yi.pow(2)).clamp(min=1e-7))
        cos_theta_yi_mod = (
            cos_theta_yi * torch.cos(adaptive_margin)
            - sin_theta_yi * torch.sin(adaptive_margin)
        )

        logits = self.s * cos_theta.clone()
        logits[batch_indices, y_true] = cos_theta_yi_mod * self.s

        return F.cross_entropy(logits, y_true)