"""Conformal Mondrian sobre el simplex. Spec: specs/70 - Hito 7.

Post-hoc (corre sobre logits guardados, sin reentrenar). Estratos: clase x
bins de N_eff (y/o radio). Calibrar en MaNGIA-val; en MaNGA la
intercambiabilidad se rompe (covariate shift): reportar cobertura EMPIRICA
en manga_gz3d_val y discutir la perdida de garantia - sin overclaim.
"""
from __future__ import annotations

import math

import torch


def make_strata(Y_soft: torch.Tensor, n_eff: torch.Tensor,
                n_eff_bins: tuple[float, ...] = (25.0, 100.0)) -> torch.Tensor:
    """Estrato = clase_verdadera * (n_bins+1) + bin de N_eff. (B, H, W) long."""
    cls = Y_soft.argmax(1)
    bins = torch.bucketize(n_eff, torch.tensor(n_eff_bins,
                                               device=n_eff.device))
    return cls * (len(n_eff_bins) + 1) + bins


def calibrate_mondrian(probs_cal: torch.Tensor, Y_cal: torch.Tensor,
                       mask_cal: torch.Tensor, strata: torch.Tensor,
                       alpha: float = 0.1) -> dict[int, float]:
    """Umbral por estrato para cobertura condicional 1-alpha.

    Score de no-conformidad: s = 1 - p_y (y = argmax de Y_soft del spaxel
    de calibracion). Cuantil con correccion de muestra finita
    ceil((n+1)(1-alpha))/n. Estratos con <10 spaxels caen al umbral global.
    """
    y_idx = Y_cal.argmax(1, keepdim=True)
    s_all = 1.0 - probs_cal.gather(1, y_idx).squeeze(1)
    s_valid = s_all[mask_cal]
    strata_valid = strata[mask_cal]

    def _quantile(s: torch.Tensor) -> float:
        n = len(s)
        q = min(math.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        return float(torch.quantile(s.float(), q))

    q_global = _quantile(s_valid)
    q: dict[int, float] = {-1: q_global}
    for g in strata_valid.unique().tolist():
        s_g = s_valid[strata_valid == g]
        q[int(g)] = _quantile(s_g) if len(s_g) >= 10 else q_global
    return q


def predict_sets(probs: torch.Tensor, strata: torch.Tensor,
                 q: dict[int, float]) -> torch.Tensor:
    """Conjunto por spaxel: {c : prob_c >= 1 - q[estrato]}. (B, K, H, W) bool.
    Cobertura 1-alpha garantizada (intercambiabilidad) por estrato."""
    thr = torch.full_like(strata, q[-1], dtype=probs.dtype)
    for g, qg in q.items():
        if g >= 0:
            thr[strata == g] = qg
    sets = probs >= (1.0 - thr).unsqueeze(1)
    # el conjunto nunca es vacio: incluir siempre el argmax
    top = probs.argmax(1, keepdim=True)
    return sets.scatter(1, top, True)


def coverage_efficiency(sets: torch.Tensor, Y_soft: torch.Tensor,
                        mask: torch.Tensor) -> tuple[float, float]:
    """(cobertura de la clase verdadera, tamano medio del conjunto)."""
    y_idx = Y_soft.argmax(1, keepdim=True)
    cov = sets.gather(1, y_idx).squeeze(1)[mask].float().mean()
    size = sets.sum(1)[mask].float().mean()
    return float(cov), float(size)
