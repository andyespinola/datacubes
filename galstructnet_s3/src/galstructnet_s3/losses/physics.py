"""L_phys ponderada por masa (correccion C7). Spec: specs/50_loss.md - Hito 4.

IMPLEMENTACION DE REFERENCIA transcrita del spec.
"""
from __future__ import annotations
import torch


def physics_constraint_loss(prob, mask, w_map, target_fractions,
                            tol: float = 0.05) -> torch.Tensor:
    """Fracciones DE MASA (peso w_map = masa raw) vs fracciones del catalogo."""
    w = (w_map * mask.float()).unsqueeze(1)                       # (B,1,H,W)
    frac = (prob * w).sum((2, 3)) / w.sum((1, 2, 3)).clamp_min(1e-6).unsqueeze(1)
    return ((frac - target_fractions).abs() - tol).clamp_min(0).mean()
