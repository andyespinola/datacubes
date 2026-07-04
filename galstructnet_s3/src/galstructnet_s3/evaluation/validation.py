"""Validacion externa y curvas de robustez. Spec: specs/70 - Hito 7.

- compare_with_gz3d: acuerdo bar/arm SOLO sobre manga_gz3d_val (disjunto
  del weak usado en L_weak; assert espejo del de training).
- Barrido de ruido y dropout de spaxels: los dos experimentos firma de la
  familia EPN (specs/45 'Plan de ablacion') - degradar la entrada
  ACTUALIZANDO c; una red estandar no tiene como saberlo.
"""
from __future__ import annotations

from typing import Callable

import torch

from ..data.dataset import check_gz3d_partition
from .metrics import compute_iou_per_class, soft_nll

BAR_IDX, ARM_IDX = 2, 3


@torch.no_grad()
def compare_with_gz3d(model, loader, gz3d_root: str, device: str = "cpu",
                      bar_idx: int = BAR_IDX, arm_idx: int = ARM_IDX) -> dict:
    """Acuerdo de prob_bar/prob_arm (plano observado) con mascaras GZ3D.

    El loader debe servir SOLO galaxias de manga_gz3d_val con claves
    gz3d_frac/gz3d_mask (como en L_weak). Metricas: IoU bar/arm con umbral
    0.5 y correlacion de Pearson con la fraccion de voto.
    """
    check_gz3d_partition(gz3d_root)                # espejo del assert de 60
    model.eval()
    ious_bar: list[float] = []
    ious_arm: list[float] = []
    corrs: list[float] = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        out = model(batch)
        prob = out["prob_obs_lum"]
        for ch, idx, acc in ((0, bar_idx, ious_bar), (1, arm_idx, ious_arm)):
            cover = batch["gz3d_mask"][:, ch]
            pred = (prob[:, idx] > 0.5) & cover
            true = (batch["gz3d_frac"][:, ch] > 0.5) & cover
            union = (pred | true).sum()
            if union > 0:
                acc.append(float((pred & true).sum() / union))
            p = prob[:, idx][cover]
            t = batch["gz3d_frac"][:, ch][cover]
            if len(p) > 2 and p.std() > 0 and t.std() > 0:
                corrs.append(float(torch.corrcoef(
                    torch.stack([p, t]))[0, 1]))
    def _mean(x): return float(torch.tensor(x).mean()) if x else float("nan")
    return {"iou_bar": _mean(ious_bar), "iou_arm": _mean(ious_arm),
            "vote_corr": _mean(corrs)}


def degrade_batch_noise(batch: dict, factor: float,
                        rng: torch.Generator | None = None) -> dict:
    """Barrido de ruido en test (specs/45): anade ruido consistente con la
    IVAR escalada x`factor` Y ACTUALIZA c (sigma_new^2 = factor * sigma^2
    => c' = c / (factor + (1-factor)*c), de c = s2r/(s2+s2r)).

    factor=1 es identidad. EPN debe degradar gracefully; A0 no puede
    saber que la entrada empeoro.
    """
    out = dict(batch)
    if factor <= 1.0:
        return out
    extra = factor - 1.0
    for sig_key, c_key in (("maps", "c_phys"), ("image", "c_spat"),
                           ("cube", "c_spec")):
        x = out[sig_key]
        c = out[c_key].clamp(0.0, 1.0)
        # sigma^2 en unidades de sigma_ref^2: s2 = (1-c)/c
        s2 = (1.0 - c) / c.clamp_min(1e-6)
        noise = (torch.randn(x.shape, generator=rng, dtype=x.dtype)
                 * torch.sqrt(extra * s2))       # broadcast: c_spec es (1,H,W)
        out[sig_key] = torch.where(c > 0, x + noise, x)
        c_new = c / (factor + (1.0 - factor) * c).clamp_min(1e-6)
        out[c_key] = torch.where(c > 0, c_new.clamp(0.0, 1.0), c)
    return out


def degrade_batch_dropout(batch: dict, frac: float,
                          rng: torch.Generator | None = None) -> dict:
    """Dropout de spaxels (specs/45): apaga aleatoriamente `frac` de los
    spaxels validos poniendo c=0 en TODAS las modalidades (la senal queda:
    c=0 basta). El baseline estandar los ve como valores crudos."""
    out = dict(batch)
    M = batch["M"]
    drop = (torch.rand(M.shape, generator=rng) < frac) & M
    for c_key in ("c_phys", "c_spat", "c_spec"):
        out[c_key] = out[c_key].masked_fill(drop.unsqueeze(1), 0.0)
    out["dropped"] = drop
    return out


@torch.no_grad()
def robustness_sweep(model, batch: dict,
                     degrade: Callable[[dict, float], dict],
                     levels: tuple[float, ...],
                     device: str = "cpu") -> dict[float, dict]:
    """Curva metrica-vs-nivel para un batch (se agrega fuera por split).
    Devuelve {nivel: {soft_nll, iou_med}} para el target de luz."""
    model.eval()
    results: dict[float, dict] = {}
    for lvl in levels:
        b = degrade(dict(batch), lvl)
        b = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in b.items()}
        out = model(b)
        ms = b["M"] & ~b["M_unc_lum"]
        ious = compute_iou_per_class(out["lum"]["prob"], b["Y_lum"], ms)
        results[lvl] = {
            "soft_nll": float(soft_nll(out["lum"]["prob"], b["Y_lum"], ms)),
            "iou_med": float(ious[~ious.isnan()].median())
            if (~ious.isnan()).any() else float("nan"),
        }
    return results
