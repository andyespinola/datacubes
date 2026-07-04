"""Supervision Dirichlet anclada en N_eff. Spec: specs/50_loss.md - Hito 4.

IMPLEMENTACION DE REFERENCIA transcrita del spec (nucleo de la innovacion).
Computar SIEMPRE en float32 (autocast off local) - lgamma/digamma en bf16
pierden precision con alpha grandes.
"""
from __future__ import annotations
import torch


def build_anchor(Y: torch.Tensor, n_eff: torch.Tensor, kappa: float = 0.5,
                 n_eff_cap: float | None = None, alpha0: float = 1.0) -> torch.Tensor:
    """alpha* = kappa * N_eff * Y + alpha0. Cap en p99(train) recomendado."""
    if n_eff_cap is not None:
        n_eff = n_eff.clamp_max(n_eff_cap)
    return kappa * n_eff.unsqueeze(1) * Y + alpha0


def dirichlet_kl(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """KL( Dir(a) || Dir(b) ), forma cerrada, enmascarada y promediada."""
    with torch.autocast(device_type=a.device.type, enabled=False):
        a, b = a.float(), b.float()
        a0, b0 = a.sum(1, keepdim=True), b.sum(1, keepdim=True)
        kl = (torch.lgamma(a0) - torch.lgamma(a).sum(1, keepdim=True)
              - torch.lgamma(b0) + torch.lgamma(b).sum(1, keepdim=True)
              + ((a - b) * (torch.digamma(a) - torch.digamma(a0)))
              .sum(1, keepdim=True)).squeeze(1)
        m = mask.float()
        return (kl * m).sum() / m.sum().clamp_min(1.0)


def anchored_seg_loss(alpha_pred, Y_raw, n_eff_raw, mask, kappa: float = 0.5,
                      n_eff_cap: float | None = None,
                      direction: str = "forward") -> torch.Tensor:
    assert (alpha_pred >= 1.0).all(), "alpha < 1: revisar cabeza (digamma inestable)"
    a_star = build_anchor(Y_raw, n_eff_raw, kappa, n_eff_cap)
    if direction == "forward":
        return dirichlet_kl(a_star, alpha_pred, mask)
    return dirichlet_kl(alpha_pred, a_star, mask)
