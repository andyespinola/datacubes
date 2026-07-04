"""Combinadora GalStructNetLossV3. Spec: specs/50_loss.md - Hito 4.

L_total = sum_t lambda_t * [L_seg(t) + beta*L_dice(t) + delta*L_PSF(t)]
        + alpha*L_boundary + eps*L_phys   (+ consist/weak: Trainer, Etapa 3+)

El regularizador KL-a-uniforme de v2 fue eliminado (C2). Mascaras:
mask_strict = M & ~M_unc_t (seg/dice/PSF); mask_loose = M (boundary/phys).
Cada termino se devuelve por separado (wandb los loguea individualmente).
"""
from __future__ import annotations

import torch
from torch import nn

from ..models.heads.boundary import boundary_loss
from .dice import dice_loss_multiclass
from .dirichlet import anchored_seg_loss
from .physics import physics_constraint_loss
from .psf import psf_loss


class GalStructNetLossV3(nn.Module):
    def __init__(self, w_seg: float = 1.0, w_dice: float = 0.5,
                 w_boundary: float = 0.3, w_psf: float = 0.4,
                 w_phys: float = 0.1, lambda_mass: float = 0.3,
                 kappa: float = 0.5, n_eff_cap: float | None = None,
                 kl_direction: str = "forward", psf_mode: str = "evidence",
                 boundary_tau: float = 0.1, phys_tol: float = 0.05):
        super().__init__()
        self.w = {"seg": w_seg, "dice": w_dice, "boundary": w_boundary,
                  "psf": w_psf, "phys": w_phys}
        self.lambda_t = {"lum": 1.0, "mass": lambda_mass}
        self.kappa = kappa
        self.n_eff_cap = n_eff_cap
        self.kl_direction = kl_direction
        self.psf_mode = psf_mode
        self.boundary_tau = boundary_tau
        self.phys_tol = phys_tol

    def forward(self, outputs: dict, batch: dict) -> dict:
        L: dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=batch["M"].device)

        for t, lam in self.lambda_t.items():
            ms = batch["M"] & ~batch[f"M_unc_{t}"]          # mask_strict
            L[f"seg_{t}"] = anchored_seg_loss(
                outputs[t]["alpha"], batch[f"Y_{t}"], batch[f"n_eff_{t}"],
                ms, self.kappa, self.n_eff_cap, self.kl_direction)
            L[f"dice_{t}"] = dice_loss_multiclass(
                outputs[t]["prob"], batch[f"Y_{t}"], ms)
            total = total + lam * (self.w["seg"] * L[f"seg_{t}"]
                                   + self.w["dice"] * L[f"dice_{t}"])
            has_psf = (f"alpha_obs_{t}" in outputs
                       or f"prob_obs_{t}" in outputs)
            if self.w["psf"] > 0 and has_psf:
                L[f"psf_{t}"] = psf_loss(outputs, batch, t, ms,
                                         self.psf_mode, self.kappa,
                                         self.n_eff_cap, self.kl_direction)
                total = total + lam * self.w["psf"] * L[f"psf_{t}"]

        if self.w["boundary"] > 0:
            # fronteras cientificas: solo target de luz (specs/50)
            L["boundary"] = boundary_loss(outputs["boundary"],
                                          batch["Y_lum"], batch["M"],
                                          self.boundary_tau)
            total = total + self.w["boundary"] * L["boundary"]

        if self.w["phys"] > 0:
            L["phys"] = physics_constraint_loss(
                outputs["lum"]["prob"], batch["M"], batch["w_phys_mass"],
                batch["target_fractions_mass"], self.phys_tol)
            total = total + self.w["phys"] * L["phys"]

        L["total"] = total
        return L
