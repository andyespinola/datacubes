"""P3 PrecisionGatedFusion. Spec: specs/23_fusion.md + specs/45 P3 - Hito 4.

Por posicion, M=3 tokens (spat/spec/phys); logits sesgados por g_m(log c_m)
con g_m init->0 (neutralidad inicial). Devuelve (F_fused (B,384,H,W),
c_fused (B,1,H,W), attn_w (B,3,H,W)). attn_w se exporta SIEMPRE en eval
(producto cientifico: "en que modalidad se apoyo el modelo").

La atencion es multi-cabeza implementada a mano: sobre M=3 keys por
posicion el costo es O(H*W*M^2) y el control de attn_mask/pesos es exacto
(el broadcasting de nn.MultiheadAttention no aporta aqui; specs/23 notas).
El mismo sesgo se aplica a todas las cabezas (specs/45 notas).
"""
from __future__ import annotations

import torch
from torch import nn

MODS = ("spat", "spec", "phys")


def _make_gate() -> nn.Sequential:
    """MLP 1->16->1 con salida inicializada a 0: al inicio el gating no
    perturba (equivale a atencion sin precision; test de neutralidad)."""
    mlp = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 1))
    last = mlp[-1]
    assert isinstance(last, nn.Linear)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return mlp


def log_clamped(c: torch.Tensor) -> torch.Tensor:
    """log(c) acotado: certeza 0 no produce -inf (el -9.2 ya sesga)."""
    return torch.log(c.clamp_min(1e-4))


class PrecisionGatedFusion(nn.Module):
    def __init__(self, d_spat: int = 256, d_spec: int = 256, d_phys: int = 64,
                 d: int = 384, n_heads: int = 4):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.n_heads = n_heads
        self.proj = nn.ModuleDict({
            "spat": nn.Conv2d(d_spat, d, 1),
            "spec": nn.Conv2d(d_spec, d, 1),
            "phys": nn.Conv2d(d_phys, d, 1)})
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.gate = nn.ModuleDict({m: _make_gate() for m in MODS})
        self.out = nn.Conv2d(d, d, 1)
        self.norm = nn.GroupNorm(8, d)

    def forward(self, F_spat, F_spec, F_phys, cbar: dict):
        B, _, H, W = F_spat.shape
        HW = H * W
        dh = self.d // self.n_heads

        toks = {m: self.proj[m](f) for m, f in
                zip(MODS, (F_spat, F_spec, F_phys))}
        # (B, d, H, W) -> (B, HW, M, d): cada posicion es una "frase" M=3
        seq = torch.stack([toks[m].flatten(2).transpose(1, 2)
                           for m in MODS], dim=2)

        # sesgo aditivo de precision en los logits (mismo para las cabezas)
        bias = torch.stack(
            [self.gate[m](log_clamped(cbar[m]).flatten(2).transpose(1, 2))
             for m in MODS], dim=2).squeeze(-1)              # (B, HW, M)

        q = self.q_proj(seq[:, :, 0])                        # query = spat
        k = self.k_proj(seq)
        v = self.v_proj(seq)
        q = q.view(B, HW, self.n_heads, dh)
        k = k.view(B, HW, 3, self.n_heads, dh)
        v = v.view(B, HW, 3, self.n_heads, dh)
        logits = torch.einsum("bnhd,bnmhd->bnhm", q, k) / dh ** 0.5
        logits = logits + bias.unsqueeze(2)                  # broadcast heads
        w_heads = logits.softmax(dim=-1)                     # (B, HW, h, M)
        fused = torch.einsum("bnhm,bnmhd->bnhd", w_heads, v).reshape(B, HW,
                                                                     self.d)
        F_f = fused.transpose(1, 2).view(B, self.d, H, W)

        w = w_heads.mean(dim=2)                              # (B, HW, M)
        cstack = torch.stack([cbar[m].flatten(2).squeeze(1) for m in MODS],
                             dim=2)                          # (B, HW, M)
        c_f = (w * cstack).sum(dim=2).view(B, 1, H, W)       # promedio convexo

        F_f = self.norm(self.out(F_f) + toks["spat"])        # residual spat
        attn_w = w.transpose(1, 2).view(B, 3, H, W)
        return F_f, c_f, attn_w
