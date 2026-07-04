"""Encoder fisico. Spec: specs/22_encoder_physical.md - Hito 3 (Std) / 4 (N, MP).

8 canales: v, sigma, age, Z, mass, av, h3, h4 (h3/h4 de pPXF - distincion
factual). Tres variantes con interfaz uniforme:
  forward(maps (B,8,H,W), c_phys (B,8,H,W)) -> (features (B,64,H,W), c_out (B,1,H,W))
"""
from __future__ import annotations

import torch
from torch import nn


class PhysicalEncoderStd(nn.Module):
    """Baseline A0: CNN de 3 capas Conv3x3 + GroupNorm + GELU, sin pooling.

    Ignora c_phys salvo para anular senal (x = x * (c > 0), equivalente
    funcional del nan_to_num v2); c_out = avg(c_phys) para interfaz uniforme.
    """

    def __init__(self, in_ch: int = 8, d_out: int = 64, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden * 2, 3, padding=1),
            nn.GroupNorm(8, hidden * 2), nn.GELU(),
            nn.Conv2d(hidden * 2, d_out, 3, padding=1),
            nn.GroupNorm(8, d_out), nn.GELU(),
        )

    def forward(self, maps: torch.Tensor, c_phys: torch.Tensor):
        x = maps * (c_phys > 0)
        return self.net(x), c_phys.mean(dim=1, keepdim=True)


class PhysicalEncoderN(nn.Module):
    """Default A1+: pila NormConv2d (specs/45 P1). Garantias heredadas:
    spaxels con c=0 no contaminan vecinos; bordes del FOV con certeza menor
    automatica; equivariancia D4 exacta."""

    def __init__(self, in_ch: int = 8, d_out: int = 64, hidden: int = 32):
        super().__init__()
        from ..layers.normconv import NormConv2d
        self.l1 = NormConv2d(in_ch, hidden, k=3)
        self.l2 = NormConv2d(hidden, hidden * 2, k=3)
        self.l3 = NormConv2d(hidden * 2, d_out, k=1)

    def forward(self, maps: torch.Tensor, c_phys: torch.Tensor):
        x, c = self.l1(maps, c_phys)
        x, c = self.l2(x, c)
        x, c = self.l3(x, c)
        return x, c.mean(dim=1, keepdim=True)


class PhysicalEncoderMP(nn.Module):
    """Nivel B (opcional): propagacion analitica de momentos (mu, var)."""

    def __init__(self, in_ch: int = 8, d_out: int = 64):
        super().__init__()
        raise NotImplementedError("experimento lateral - specs/22 'Nivel B'")
