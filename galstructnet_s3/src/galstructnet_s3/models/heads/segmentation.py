"""Cabezas Dirichlet duales. Spec: specs/40_head_segmentation.md - Hito 4.

EvidenceHead (specs/45 P4): e = softplus(proj(hidden)) * h(c_dec),
h(u) = softplus(a*log u + b); alpha = 1 + e. Dos cabezas (mass/lum, C3);
(a, b) persistidos con nombre en el checkpoint. Baseline A0: DirichletSegHead
(softplus directo, sin techo de certeza), tambien dual.

Salida por target (specs/40 'Contratos'):
    {alpha (B,K,H,W) >= 1, prob = alpha/S, evidence = alpha-1, vacuity = K/S}
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _dirichlet_outputs(e: torch.Tensor) -> dict:
    alpha = e + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    return {"alpha": alpha, "prob": alpha / S, "evidence": e,
            "vacuity": alpha.shape[1] / S}


class DirichletSegHead(nn.Module):
    """Baseline A0 (EDL clasica anclada): alpha = 1 + softplus(proj(hidden)).
    Ignora c_dec (sin techo de certeza) - ese es el punto del baseline."""

    def __init__(self, in_ch: int = 256, n_classes: int = 5):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, n_classes, 1)

    def forward(self, hidden: torch.Tensor, c_dec: torch.Tensor) -> dict:
        return _dirichlet_outputs(F.softplus(self.proj(hidden)))


class EvidenceHead(nn.Module):
    """P4 (specs/45): e = e_net * h(c_dec), h(u) = softplus(a*log u + b).

    El techo de evidencia escala con la certeza propagada; la red puede
    emitir menos evidencia que el techo (regiones de feature-space no
    vistas), nunca mas sin respaldo instrumental. c_dec -> 0 => alpha -> 1
    => vacuity = 1: ignorancia instrumental => ignorancia del modelo.

    (a, b) se persisten con nombre en el checkpoint: son los dos numeros
    que el capitulo de la tesis interpreta. Init de b: calibrar con
    scripts/init_evidence_scale.py para que S_init ~ K + kappa*mediana(N_eff).
    """

    def __init__(self, in_ch: int = 256, n_classes: int = 5,
                 b_init: float = 4.0):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, n_classes, 1)
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(b_init))

    def forward(self, hidden: torch.Tensor, c_dec: torch.Tensor) -> dict:
        e_net = F.softplus(self.proj(hidden))
        # en c_dec == 0 exacto, h se fuerza a 0 (el clamp del log dejaria
        # h(1e-6) > 0 y romperia la garantia alpha == 1)
        h = F.softplus(self.a * torch.log(c_dec.clamp_min(1e-6)) + self.b)
        h = torch.where(c_dec > 0, h, torch.zeros_like(h))
        return _dirichlet_outputs(e_net * h)


class DualSegHeads(nn.Module):
    """Tronco compartido, dos proyecciones finales (C3). La discrepancia
    JS(p_mass || p_lum) por spaxel se calcula en evaluation (specs/40)."""

    def __init__(self, in_ch: int = 256, n_classes: int = 5,
                 share_gate: bool = True, kind: str = "evidence",
                 b_init: float = 4.0):
        super().__init__()
        if kind == "evidence":
            self.head_mass: nn.Module = EvidenceHead(in_ch, n_classes, b_init)
            self.head_lum: nn.Module = EvidenceHead(in_ch, n_classes, b_init)
        else:
            self.head_mass = DirichletSegHead(in_ch, n_classes)
            self.head_lum = DirichletSegHead(in_ch, n_classes)
        if kind == "evidence" and share_gate:      # (a, b) compartidos (ablar)
            self.head_lum.a = self.head_mass.a
            self.head_lum.b = self.head_mass.b

    def forward(self, hidden: torch.Tensor, c_dec: torch.Tensor) -> dict:
        return {"mass": self.head_mass(hidden, c_dec),
                "lum": self.head_lum(hidden, c_dec)}
