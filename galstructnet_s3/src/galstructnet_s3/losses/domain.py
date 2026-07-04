"""Adaptacion de dominio (Etapa 4). Spec: specs/60 'Etapa 4' - Hito 6.

MMD primero (mas simple/estable); DANN queda como plan B (ADR si se
necesita). Self-training solo como ultimo recurso (filtrado por S alto y
conjunto conformal unitario, auditado contra GZ3D holdout).
"""
from __future__ import annotations

import torch


def _rbf_kernel(a: torch.Tensor, b: torch.Tensor,
                scales: tuple[float, ...]) -> torch.Tensor:
    d2 = torch.cdist(a, b) ** 2
    k = torch.zeros_like(d2)
    for s in scales:
        k = k + torch.exp(-d2 / (2 * s ** 2))
    return k / len(scales)


def mmd_loss(feats_a: torch.Tensor, feats_b: torch.Tensor,
             scales: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)) -> torch.Tensor:
    """MMD^2 multi-kernel RBF entre embeddings (N_a, D) y (N_b, D).

    Los embeddings son el `hidden` del decoder global-avg-pooleado por
    galaxia (o por spaxel submuestreado). Bandwidths fijos multi-escala
    (estandar; evitar el median heuristic por estabilidad en batches chicos).
    """
    if feats_a.numel() == 0 or feats_b.numel() == 0:
        return torch.zeros((), device=feats_a.device)
    k_aa = _rbf_kernel(feats_a, feats_a, scales)
    k_bb = _rbf_kernel(feats_b, feats_b, scales)
    k_ab = _rbf_kernel(feats_a, feats_b, scales)
    return k_aa.mean() + k_bb.mean() - 2 * k_ab.mean()


def pooled_embeddings(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, C): promedio sobre spaxels validos."""
    m = mask.float().unsqueeze(1)
    return (hidden * m).sum((2, 3)) / m.sum((2, 3)).clamp_min(1.0)
