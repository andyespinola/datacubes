"""Regularizador de fronteras (NO es una cabeza: sin parametros).

Spec: specs/41_head_boundary.md (reframing v3) - Hito 4. B = exp(-|grad p|/tau),
tau=0.1, sobre prob_lum; L_boundary = MSE(B(prob), B(Y_lum_raw)) en M_valid.
Implementacion de referencia (diferencias finitas); validar contra tests.
"""
from __future__ import annotations
import torch


def boundary_map(prob: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """prob (B,K,H,W) -> B (B,1,H,W) en (0,1]; 1 = interior, ->0 = frontera."""
    dy = prob.diff(dim=-2, prepend=prob[..., :1, :])
    dx = prob.diff(dim=-1, prepend=prob[..., :, :1])
    g = torch.sqrt((dy ** 2 + dx ** 2).sum(dim=1, keepdim=True).clamp_min(1e-12))
    return torch.exp(-g / tau)


def boundary_loss(prob: torch.Tensor, Y: torch.Tensor, mask: torch.Tensor,
                  tau: float = 0.1) -> torch.Tensor:
    m = mask.float().unsqueeze(1)
    d = (boundary_map(prob, tau) - boundary_map(Y, tau)) ** 2
    return (d * m).sum() / m.sum().clamp_min(1.0)
