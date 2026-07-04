"""Fusiones estandar: concat (A2-a) y cross-attention global (A2-b / A0).

Spec: specs/23_fusion.md 'Variantes' - Hito 3. Interfaz uniforme con
fusion_precision (A2-c, Hito 4):

    forward(F_spat (B,256,H,W), F_spec (B,256,H,W), F_phys (B,64,H,W),
            cbar {spat,spec,phys: (B,1,H,W)})
      -> (F_fused (B,384,H,W), c_fused (B,1,H,W), attn_w (B,3,H,W) | None)

`concat` y `global` calculan c_fused como promedio simple de cbar (specs/23).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

MODS = ("spat", "spec", "phys")


def _cbar_mean(cbar: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([cbar[m] for m in MODS], dim=1).mean(dim=1, keepdim=True)


class ConcatFusion(nn.Module):
    """A2-a: concat + Conv1x1 (baseline minimo, la opcion 1 de v2)."""

    def __init__(self, d_spat: int = 256, d_spec: int = 256, d_phys: int = 64,
                 d: int = 384):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(d_spat + d_spec + d_phys, d, 1),
            nn.GroupNorm(8, d), nn.GELU())

    def forward(self, F_spat, F_spec, F_phys, cbar):
        fused = self.proj(torch.cat([F_spat, F_spec, F_phys], dim=1))
        return fused, _cbar_mean(cbar), None


class CrossAttentionFusion(nn.Module):
    """A2-b (fusion del A0): atencion global de v2, migrada con la interfaz
    v3. Query = tokens espaciales de F_spat; K/V = tokens de las tres
    modalidades proyectadas a d. Usa scaled_dot_product_attention (flash
    cuando esta disponible) y residual con F_spat (specs/23 'Variantes').

    Nota honesta (specs/23 'Por que cambia'): esta variante mezcla gating de
    modalidad con contexto espacial; se conserva SOLO como ablacion/baseline.
    """

    def __init__(self, d_spat: int = 256, d_spec: int = 256, d_phys: int = 64,
                 d: int = 384, n_heads: int = 4):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.proj = nn.ModuleDict({
            "spat": nn.Conv2d(d_spat, d, 1),
            "spec": nn.Conv2d(d_spec, d, 1),
            "phys": nn.Conv2d(d_phys, d, 1)})
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out = nn.Conv2d(d, d, 1)
        self.norm = nn.GroupNorm(8, d)

    def forward(self, F_spat, F_spec, F_phys, cbar):
        B, _, H, W = F_spat.shape
        toks = {m: self.proj[m](f) for m, f in
                zip(MODS, (F_spat, F_spec, F_phys))}
        spat_seq = toks["spat"].flatten(2).transpose(1, 2)      # (B, HW, d)
        kv_seq = torch.cat([toks[m].flatten(2).transpose(1, 2)
                            for m in MODS], dim=1)              # (B, 3HW, d)

        def split_heads(x):
            return x.view(B, -1, self.n_heads,
                          self.d // self.n_heads).transpose(1, 2)

        q = split_heads(self.q_proj(spat_seq))
        k = split_heads(self.k_proj(kv_seq))
        v = split_heads(self.v_proj(kv_seq))
        att = F.scaled_dot_product_attention(q, k, v)           # flash si hay
        att = att.transpose(1, 2).reshape(B, H * W, self.d)
        fused = att.transpose(1, 2).view(B, self.d, H, W)
        fused = self.norm(self.out(fused) + toks["spat"])       # residual spat
        return fused, _cbar_mean(cbar), None
