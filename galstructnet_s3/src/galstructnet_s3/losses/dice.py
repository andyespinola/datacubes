"""Dice multiclase suave (codigo v2). Spec: specs/50 'L_dice' - Hito 4.

Contrapeso del desbalance de clases (barra/brazo 5-10% de spaxels).
"""
from __future__ import annotations

import torch


def dice_loss_multiclass(prob: torch.Tensor, Y: torch.Tensor,
                         mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 - Dice suave promediado sobre clases, en spaxels enmascarados.

    prob, Y: (B, K, H, W); mask: (B, H, W) bool.
    """
    m = mask.float().unsqueeze(1)
    inter = (prob * Y * m).sum(dim=(0, 2, 3))
    denom = ((prob + Y) * m).sum(dim=(0, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()
