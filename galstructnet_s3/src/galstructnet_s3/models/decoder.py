"""Decoder FPN top-down. Spec: specs/30_decoder_unet.md - Hito 3 (std) / 4 (N).

Tamanos SIEMPRE de features.shape[-2:] (C5). Variante NormConv: bloques
(senal, certeza); skips Swin reciben c via avg-pool de c_spat por escala.

Contrato (interfaz uniforme para std y N):
    forward(features (B,384,H,W), skips [(B,256,H_i,W_i)]x3,
            c_fused (B,1,H,W), c_skips opcional)
      -> (hidden (B,256,H,W), c_dec (B,1,H,W))
En la variante std, c_dec := c_fused pasa-traves (specs/30).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _block(i: int, o: int, n: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    for j in range(n):
        layers += [nn.Conv2d(i if j == 0 else o, o, 3, padding=1),
                   nn.GroupNorm(8, o), nn.GELU()]
    return nn.Sequential(*layers)


class FPNDecoder(nn.Module):
    """features ya esta a alta resolucion; cada skip se upsamplea al tamano
    de features y se mezcla con un bloque Conv-GN-GELU x2 (specs/30 A0)."""

    def __init__(self, in_ch: int = 384, skip_ch=(256, 256, 256),
                 out_ch: int = 256, mid: int = 256):
        super().__init__()
        self.in_proj = nn.Conv2d(in_ch, mid, 1)
        self.blocks = nn.ModuleList([_block(mid + s, mid) for s in skip_ch])
        self.final = _block(mid, out_ch, n=3)

    def forward(self, features: torch.Tensor, skips: list[torch.Tensor],
                c_fused: torch.Tensor, c_skips=None):
        hw = features.shape[-2:]                      # dinamico, sin literales
        x = self.in_proj(features)
        for skip, block in zip(skips, self.blocks):   # profundo -> superficial
            s = F.interpolate(skip, size=hw, mode="bilinear",
                              align_corners=False)
            x = block(torch.cat([x, s], dim=1))
        return self.final(x), c_fused                 # c_dec pasa-traves (std)


class _NormBlock(nn.Module):
    """n capas NormConv2d encadenadas sobre (x, c)."""

    def __init__(self, i: int, o: int, n: int = 2):
        super().__init__()
        from .layers.normconv import NormConv2d
        self.layers = nn.ModuleList(
            [NormConv2d(i if j == 0 else o, o, k=3) for j in range(n)])

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        for layer in self.layers:
            x, c = layer(x, c)
        return x, c


class FPNDecoderN(nn.Module):
    """Variante A1 (specs/30): mismos bloques con NormConv2d; cada bloque
    consume y emite (x, c); las skips entran como (skip, c_skip upsampleada)
    concatenadas en senal y certeza. c_dec = certeza del bloque final.

    Aproximacion documentada: las skips vienen del Swin (estandar, sin
    certeza propia); c_skips = avg-pool de c_spat por escala.
    """

    def __init__(self, in_ch: int = 384, skip_ch=(256, 256, 256),
                 out_ch: int = 256, mid: int = 256):
        super().__init__()
        self.in_proj = nn.Conv2d(in_ch, mid, 1)
        self.blocks = nn.ModuleList(
            [_NormBlock(mid + s, mid) for s in skip_ch])
        self.final = _NormBlock(mid, out_ch, n=3)

    @staticmethod
    def _up_normalized(s: torch.Tensor, cs: torch.Tensor, hw,
                       eps: float = 1e-6):
        """Upsampling normalizado por certeza: interp(s*c)/interp(c).
        Interpolar (s, c) por separado violaria la garantia de ignorancia
        (senal con c=0 se colaria en pixeles vecinos via bilinear)."""
        num = F.interpolate(s * cs, size=hw, mode="bilinear",
                            align_corners=False)
        c_up = F.interpolate(cs, size=hw, mode="bilinear",
                             align_corners=False)
        return num / c_up.clamp_min(eps), c_up

    def forward(self, features: torch.Tensor, skips: list[torch.Tensor],
                c_fused: torch.Tensor, c_skips: list[torch.Tensor] | None = None):
        hw = features.shape[-2:]
        x = self.in_proj(features)
        c = c_fused.expand(-1, x.shape[1], -1, -1)
        for i, (skip, block) in enumerate(zip(skips, self.blocks)):
            cs = (c_skips[i] if c_skips is not None
                  else torch.ones(skip.shape[0], 1, *skip.shape[-2:],
                                  device=skip.device, dtype=skip.dtype))
            s, cs_up = self._up_normalized(skip, cs, hw)
            x = torch.cat([x, s], dim=1)
            c = torch.cat([c, cs_up.expand(-1, s.shape[1], -1, -1)], dim=1)
            x, c = block(x, c)
        x, c = self.final(x, c)
        return x, c.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
