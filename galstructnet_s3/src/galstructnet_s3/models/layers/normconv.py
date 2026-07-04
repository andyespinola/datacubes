"""P1 NormConv2d. Spec: specs/45_evidence_layers.md - Hito 4 (tras A0).

Contrato: forward(x (B,Cin,H,W), c (B,Cin,H,W)) -> (y (B,Cout,H,W), c_out).
Invariantes testeables: ignorancia exacta (c=0), reduccion con c==1,
certeza convexa/monotona, equivariancia D4.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class NormConv2d(nn.Module):
    """Agregacion espacial ponderada por certeza + mezcla puntual con signo.

    La aplicabilidad A es no negativa (softplus): la agregacion espacial es
    un promedio convexo ponderado por c. La expresividad con signo se
    restituye en la mezcla puntual 1x1 posterior. Las operaciones puntuales
    no alteran c (aproximacion documentada, specs/45 P1).
    """

    def __init__(self, c_in: int, c_out: int, k: int = 3, eps: float = 1e-4):
        super().__init__()
        self.theta_A = nn.Parameter(torch.randn(c_out, c_in, k, k) * 0.1)
        self.mix = nn.Conv2d(c_out, c_out, 1)
        self.norm = nn.GroupNorm(8, c_out)
        self.act = nn.GELU()
        self.eps = eps
        self.pad = k // 2

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        A = F.softplus(self.theta_A)                        # >= 0
        num = F.conv2d(x * c, A, padding=self.pad)
        den = F.conv2d(c, A, padding=self.pad)
        # con bf16, clampear den antes de dividir (specs/45 notas)
        z = num / den.clamp_min(self.eps)
        ones = torch.ones_like(c)
        c_out = F.conv2d(c, A, padding=self.pad) / \
            F.conv2d(ones, A, padding=self.pad).clamp_min(self.eps)
        y = self.act(self.norm(self.mix(z)))
        return y, c_out
