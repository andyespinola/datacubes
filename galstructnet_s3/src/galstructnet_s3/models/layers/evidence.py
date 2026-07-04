"""P2 EvidenceConv2d (redistribucion de pseudo-conteos).

Spec: specs/45_evidence_layers.md - Hito 4. Kernel estocastico >=0 suma 1;
conservacion de evidencia en el interior; con kernel fijo = PSF reproduce
la alpha-convolucion (specs/43 modo 'evidence').

Variante D (default, depthwise): conserva la suma espacial de evidencia por
clase. Variante M (mixing entre clases): solo para ablacion.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class EvidenceConv2d(nn.Module):
    """Kernel estocastico por clase: K+ >= 0, suma espacial = 1. La
    convolucion es entonces redistribucion de conteos (masa conservada en el
    interior; en bordes con zero-padding la masa que sale del FOV se pierde
    - documentado, no es bug)."""

    def __init__(self, n_ch: int, k: int,
                 fixed_kernel: torch.Tensor | None = None):
        super().__init__()
        self.n_ch = n_ch
        if fixed_kernel is not None:                        # modo PSF
            if fixed_kernel.dim() != 2:
                raise ValueError("fixed_kernel debe ser (K, K); el batching "
                                 "por muestra vive en el modulo PSF (43)")
            self.register_buffer("theta",
                                 fixed_kernel.clamp_min(1e-12).log())
            self.fixed = True
            k = fixed_kernel.shape[-1]
        else:
            self.theta = nn.Parameter(torch.zeros(n_ch, 1, k, k))
            self.fixed = False
        self.pad = k // 2

    def kernel(self) -> torch.Tensor:
        Kp = self.theta.exp() if self.fixed else F.softplus(self.theta)
        return Kp / Kp.sum(dim=(-2, -1), keepdim=True)      # estocastico

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        Kp = self.kernel().to(e.dtype)
        n_ch = e.shape[1]
        if self.fixed:
            Kp = Kp.expand(n_ch, 1, *Kp.shape[-2:])
        return F.conv2d(e, Kp, padding=self.pad, groups=n_ch)
