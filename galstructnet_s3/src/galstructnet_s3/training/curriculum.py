"""Etapas 1-4. Spec: specs/60_training.md - Hitos 5-6.

Etapa 1: masked spectral POR POSICION (SpectralMAEHead, sin pooling; C6),
datos MaNGIA U MaNGA 50/50. Etapa 2: seg+dice+phys dual. Etapa 3: +PSF,
boundary, consist, L_weak GZ3D (particionado). Etapa 4: MMD (DANN plan B).
"""
from __future__ import annotations
import torch
from torch import nn


class SpectralMAEHead(nn.Module):
    """Reconstruccion por posicion desde (N, L', d_model): proj a `patch`."""

    def __init__(self, d_model: int = 128, patch: int = 4):
        super().__init__()
        self.proj = nn.Linear(d_model, patch)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return self.proj(seq).flatten(1)


def masked_spectral_loss(rec, target, mask_lambda, mask_spatial) -> torch.Tensor:
    """MSE solo en (canal, spaxel) enmascarados Y validos (C6).

    rec/target: (B, L, H, W); mask_lambda: (L,) bool/float;
    mask_spatial: (B, H, W) bool.
    """
    diff = (rec - target) ** 2
    m = mask_lambda.unsqueeze(-1).unsqueeze(-1) * mask_spatial.unsqueeze(1).float()
    return (diff * m).sum() / m.sum().clamp_min(1.0)


class SpectralMAEPretrainer(nn.Module):
    """Etapa 1 (MAE/BERT real, C6): enmascara posiciones del eje lambda con
    un token aprendible en la ENTRADA del encoder, codifica SIN pooling
    (`return_sequence=True`) y reconstruye por posicion.

    El enmascarado es por posicion de la secuencia reducida (L' = L/patch):
    cada posicion oculta sus `patch` canales originales. La mascara se
    sortea por batch (mismo patron para todos los spaxels: la tarea es
    rellenar desde el contexto espectral, no espacial).
    """

    def __init__(self, encoder: nn.Module, mask_ratio: float = 0.30):
        super().__init__()
        if not getattr(encoder, "return_sequence", False):
            raise ValueError("el encoder debe construirse con "
                             "return_sequence=True (specs/20 parche v3)")
        self.encoder = encoder
        self.d_model = int(getattr(encoder, "d_model"))
        self.patch = int(getattr(encoder, "patch"))
        self.head = SpectralMAEHead(d_model=self.d_model, patch=self.patch)
        self.mask_token = nn.Parameter(torch.zeros(1))
        self.mask_ratio = mask_ratio

    def sample_mask(self, L: int, generator: torch.Generator | None = None
                    ) -> torch.Tensor:
        """Mascara (L,) bool a granularidad de posicion reducida."""
        patch = self.patch
        Lp = L // patch
        n_masked = max(1, int(round(self.mask_ratio * Lp)))
        idx = torch.randperm(Lp, generator=generator)[:n_masked]
        mask_p = torch.zeros(Lp, dtype=torch.bool)
        mask_p[idx] = True
        return mask_p.repeat_interleave(patch)

    def forward(self, cube: torch.Tensor, mask_spatial: torch.Tensor,
                mask_lambda: torch.Tensor | None = None) -> dict:
        B, L, H, W = cube.shape
        if mask_lambda is None:
            mask_lambda = self.sample_mask(L).to(cube.device)
        masked = torch.where(mask_lambda.view(1, L, 1, 1),
                             self.mask_token.to(cube.dtype), cube)
        seq = self.encoder(masked)                    # (B*H*W, L', d)
        rec = self.head(seq)                          # (B*H*W, L'*patch)
        rec = rec.view(B, H, W, -1).permute(0, 3, 1, 2)
        Lr = rec.shape[1]                             # == L si L%patch==0
        loss = masked_spectral_loss(rec, cube[:, :Lr],
                                    mask_lambda[:Lr].float(), mask_spatial)
        return {"loss": loss, "rec": rec, "mask_lambda": mask_lambda}


def consistency_loss(prob_a: torch.Tensor, prob_b: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """KL simetrica entre dos forwards con augmentations D4 distintas
    (ya des-rotadas y alineadas). Codigo v2 (specs/60 Etapa 3)."""
    pa = prob_a.clamp_min(1e-8)
    pb = prob_b.clamp_min(1e-8)
    kl_ab = (pa * (pa.log() - pb.log())).sum(dim=1)
    kl_ba = (pb * (pb.log() - pa.log())).sum(dim=1)
    m = mask.float()
    return (0.5 * (kl_ab + kl_ba) * m).sum() / m.sum().clamp_min(1.0)


def weak_gz3d_loss(outputs: dict, batch: dict,
                   bar_idx: int = 2, arm_idx: int = 3) -> torch.Tensor:
    """BCE de prob_bar/prob_arm (plano observado, via PSF) contra la
    fraccion de voto GZ3D, enmascarada a la cobertura (specs/60 'L_weak').

    batch:
      gz3d_frac: (B, 2, H, W) — fraccion de voto [bar, arm]
      gz3d_mask: (B, 2, H, W) bool — cobertura de cada mascara GZ3D
    Solo bar/arm: el resto del simplex queda libre.
    """
    prob_obs = outputs["prob_obs_lum"]
    pred = torch.stack([prob_obs[:, bar_idx], prob_obs[:, arm_idx]], dim=1)
    target = batch["gz3d_frac"]
    m = batch["gz3d_mask"].float()
    bce = -(target * pred.clamp_min(1e-8).log()
            + (1 - target) * (1 - pred).clamp_min(1e-8).log())
    return (bce * m).sum() / m.sum().clamp_min(1.0)
