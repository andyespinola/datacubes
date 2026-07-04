"""L_PSF anclada en el plano observado. Spec: specs/50 'L_PSF' - Hito 4.

modo 'evidence' (default con PSFEvidenceModule):
    KL( Dir(build_anchor(Y_psf, N_eff_psf)) || Dir(alpha_obs) )
modo 'prob' (baseline A4): CE suave de v2 entre prob_obs y Y_psf.
Los targets Y_*_psf / N_eff_psf_* vienen del proyector (C4): nada se
convoluciona en el DataLoader.
"""
from __future__ import annotations

import torch

from .dirichlet import build_anchor, dirichlet_kl


def psf_loss(outputs: dict, batch: dict, target: str, mask: torch.Tensor,
             mode: str = "evidence", kappa: float = 0.5,
             n_eff_cap: float | None = None,
             direction: str = "forward") -> torch.Tensor:
    Y_obs = batch[f"Y_{target}_obs"]
    if mode == "evidence":
        a_star = build_anchor(Y_obs, batch[f"n_eff_{target}_obs"],
                              kappa, n_eff_cap)
        alpha_obs = outputs[f"alpha_obs_{target}"]
        if direction == "forward":
            return dirichlet_kl(a_star, alpha_obs, mask)
        return dirichlet_kl(alpha_obs, a_star, mask)
    if mode == "prob":
        prob_obs = outputs[f"prob_obs_{target}"]
        ce = -(Y_obs * torch.log(prob_obs.clamp_min(1e-8))).sum(dim=1)
        m = mask.float()
        return (ce * m).sum() / m.sum().clamp_min(1.0)
    raise ValueError(f"psf_mode desconocido: {mode!r}")
