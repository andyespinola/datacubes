"""Descomposicion TU/AU/EU por informacion mutua. Spec: specs/42 - Hito 4.

IMPLEMENTACION DE REFERENCIA transcrita del spec (formas cerradas; C1).
Cubierta por tests/unit/test_uncertainty.py (tests 2-4 con valores exactos).
"""
from __future__ import annotations
import torch
from torch import nn


class UncertaintyDecomposition(nn.Module):
    """TU = H[alpha/S]; AU = E[H[Cat(p)]] cerrada; EU = TU - AU >= 0 (Jensen)."""

    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.K = n_classes

    def forward(self, alpha: torch.Tensor) -> dict:
        S = alpha.sum(dim=1, keepdim=True)
        p = alpha / S
        TU = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1, keepdim=True)
        AU = -(p * (torch.digamma(alpha + 1.0)
                    - torch.digamma(S + 1.0))).sum(dim=1, keepdim=True)
        EU = (TU - AU).clamp_min(0.0)  # clamp solo numerico (~1e-7)
        return {"total": TU, "aleatoric": AU, "epistemic": EU,
                "vacuity": self.K / S}
