"""Encoder espectral (Mamba). Spec: specs/20_encoder_spectral.md - Hito 3.

v3: flag return_sequence=True (Etapa 1 SSL, sin pooling); ablacion
OBLIGATORIA vs Conv1D dilatada (nivel A8); plan de cache de embeddings
post-Etapa 1. La certeza espectral entra colapsada (c_spec) via fusion.

Ambas clases comparten contrato:
    forward(cube (B,L,H,W)) -> (B,d_out,H,W)                # segmentacion
    forward(cube) con return_sequence=True -> (B*H*W,L',d_model)  # SSL C6
`mamba_ssm` se importa lazy (requiere CUDA); DilatedConv1DEncoder es el
contrafactual A8 y el camino de CI/CPU.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint


class _SpectralStem(nn.Module):
    """Reduccion inicial del eje lambda: L -> L/patch, 1 -> d_model canales.
    Compartida por Mamba y Conv1D (misma resolucion de secuencia L')."""

    def __init__(self, d_model: int, patch: int = 4):
        super().__init__()
        if patch != 4:
            raise ValueError("patch=4 es el unico stem definido (2 convs s2)")
        self.net = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N,1,L) -> (N,d,L')
        return self.net(x)


class _SpectralBase(nn.Module):
    """Esqueleto comun: stem -> bloques -> [pool + proj | secuencia]."""

    def __init__(self, d_model: int, d_out: int, patch: int,
                 return_sequence: bool, use_checkpoint: bool):
        super().__init__()
        self.d_model = d_model
        self.patch = patch
        self.return_sequence = return_sequence
        self.use_checkpoint = use_checkpoint
        self.stem = _SpectralStem(d_model, patch)
        self.norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, d_out)

    def _blocks_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        B, L, H, W = cube.shape
        x = rearrange(cube, "b l h w -> (b h w) 1 l")
        x = self.stem(x)                        # (N, d, L')
        x = rearrange(x, "n d l -> n l d")
        x = self._blocks_forward(x)
        x = self.norm(x)                        # (N, L', d)
        if self.return_sequence:
            return x                            # Etapa 1: SIN pooling (C6)
        x = self.proj_out(x.mean(dim=1))        # (N, d_out)
        return rearrange(x, "(b h w) d -> b d h w", b=B, h=H, w=W)


class BidirectionalMambaBlock(nn.Module):
    """Mamba forward + backward sumados (specs/20 'Por que bidireccional')."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        from mamba_ssm import Mamba
        self.fwd = Mamba(d_model=d_model, d_state=d_state,
                         d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=d_model, d_state=d_state,
                         d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, L, D)
        out_fwd = self.fwd(x)
        out_bwd = torch.flip(self.bwd(torch.flip(x, dims=[1])), dims=[1])
        return self.norm(x + out_fwd + out_bwd)


class MambaSpectralEncoder(_SpectralBase):
    """Cada spaxel es una secuencia espectral independiente (sin mezcla
    espacial). Gradient checkpointing por bloque (specs/20 'Memoria')."""

    def __init__(self, d_model: int = 128, d_out: int = 256, n_layers: int = 4,
                 patch: int = 4, return_sequence: bool = False,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 use_checkpoint: bool = True):
        super().__init__(d_model, d_out, patch, return_sequence, use_checkpoint)
        self.blocks = nn.ModuleList([
            BidirectionalMambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)])

    def _blocks_forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


class _DilatedBlock(nn.Module):
    """Conv1D dilatada residual (kernel 5): campo receptivo exponencial."""

    def __init__(self, d_model: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=5,
                              padding=2 * dilation, dilation=dilation)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, L, D)
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + F.gelu(y))


class DilatedConv1DEncoder(_SpectralBase):
    """Contrafactual A8 (specs/20 parche v3): misma interfaz que Mamba.
    Dilataciones 1,4,16,... : con 6 bloques cubre ~4k canales efectivos."""

    def __init__(self, d_model: int = 128, d_out: int = 256, n_layers: int = 6,
                 patch: int = 4, return_sequence: bool = False,
                 use_checkpoint: bool = False):
        super().__init__(d_model, d_out, patch, return_sequence, use_checkpoint)
        self.blocks = nn.ModuleList([
            _DilatedBlock(d_model, dilation=4 ** min(i, 5))
            for i in range(n_layers)])

    def _blocks_forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


def build_spectral_encoder(backbone: str = "mamba", **kwargs) -> nn.Module:
    """Seleccion por config (`model.spectral.backbone`, specs/20)."""
    if backbone == "mamba":
        return MambaSpectralEncoder(**kwargs)
    if backbone == "conv1d":
        return DilatedConv1DEncoder(**kwargs)
    raise ValueError(f"spectral.backbone desconocido: {backbone!r}")
