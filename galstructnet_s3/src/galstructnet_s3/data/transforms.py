"""Augmentations v3. Spec: specs/10_dataset.md - Hito 1.

Convenciones:
- Cada transform es un callable `(sample: dict, rng: np.random.Generator) -> dict`.
- `RandomDihedral` transforma TODOS los tensores espaciales juntos (senal,
  certeza, 8 tensores de etiqueta/ancla, mascaras, w_phys). PSF, escalares y
  fracciones no rotan.
- Determinismo: el dataset construye un rng por (seed, epoch, idx).
- Prohibido augmentar el eje lambda salvo el rest-frame deterministico.
"""
from __future__ import annotations

import numpy as np
import torch

# Claves espaciales del contrato (specs/10 'Contrato de salida'). Todo tensor
# con shape[-2:] == (H, W) presente en esta lista se transforma junto.
SPATIAL_KEYS = (
    "cube", "c_spec", "image", "c_spat", "maps", "c_phys",
    "Y_mass", "Y_mass_obs", "n_eff_mass", "n_eff_mass_obs",
    "Y_lum", "Y_lum_obs", "n_eff_lum", "n_eff_lum_obs",
    "M", "M_unc_mass", "M_unc_lum", "w_phys_mass",
)

C_LIGHT_KMS = 299_792.458


def apply_dihedral(sample: dict, k: int, flip: bool) -> dict:
    """Aplica un elemento de D4 (flip horizontal opcional, luego rot90 x k)
    a todos los tensores espaciales."""
    for key in SPATIAL_KEYS:
        v = sample.get(key)
        if v is None or not torch.is_tensor(v):
            continue
        if flip:
            v = torch.flip(v, dims=[-1])
        if k % 4:
            v = torch.rot90(v, k % 4, dims=(-2, -1))
        sample[key] = v
    return sample


def invert_dihedral(sample: dict, k: int, flip: bool) -> dict:
    """Inverso exacto de `apply_dihedral(k, flip)`: aplicar ambos en secuencia
    es la identidad (test 6 de specs/10)."""
    for key in SPATIAL_KEYS:
        v = sample.get(key)
        if v is None or not torch.is_tensor(v):
            continue
        if k % 4:
            v = torch.rot90(v, -(k % 4), dims=(-2, -1))
        if flip:
            v = torch.flip(v, dims=[-1])
        sample[key] = v
    return sample


class RandomDihedral:
    """Elemento aleatorio de D4 (8 simetrias) sobre todo el sample."""

    def __call__(self, sample: dict, rng: np.random.Generator) -> dict:
        k = int(rng.integers(0, 4))
        flip = bool(rng.integers(0, 2))
        return apply_dihedral(sample, k, flip)


class SpectralJitter:
    """Ruido gaussiano calibrado al ruido real por spaxel (specs/10):
    sigma proporcional a 1/snr en unidades normalizadas, via
    c_spec = snr/(snr+SNR_REF) => sigma ~ scale * (1-c)/c."""

    def __init__(self, scale: float = 0.02, sigma_max: float = 2.0):
        self.scale = scale
        self.sigma_max = sigma_max

    def __call__(self, sample: dict, rng: np.random.Generator) -> dict:
        cube, c_spec = sample["cube"], sample["c_spec"]
        c = c_spec.clamp_min(1e-3)
        sigma = (self.scale * (1.0 - c) / c).clamp_max(self.sigma_max)
        noise = torch.from_numpy(
            rng.standard_normal(cube.shape).astype(np.float32))
        sample["cube"] = cube + noise * sigma
        return sample


class ChannelDropout:
    """Apaga la CERTEZA de los canales h3/h4 (indices 6-7) con prob p.
    No toca la senal: c=0 basta (garantia P1, specs/45). Robustez a su
    ausencia en inferencia MaNGA de bajo S/N."""

    def __init__(self, p: float = 0.15, channels: tuple[int, ...] = (6, 7)):
        self.p = p
        self.channels = channels

    def __call__(self, sample: dict, rng: np.random.Generator) -> dict:
        if rng.random() < self.p:
            c_phys = sample["c_phys"].clone()
            c_phys[list(self.channels)] = 0.0
            sample["c_phys"] = c_phys
        return sample


class PSFJitter:
    """Perturbacion multiplicativa suave del kernel PSF + renormalizacion.
    Solo Etapa 4 (specs/10); el trainer la anade a la pipeline en esa etapa."""

    def __init__(self, sigma: float = 0.05):
        self.sigma = sigma

    def __call__(self, sample: dict, rng: np.random.Generator) -> dict:
        psf = sample["psf"]
        fac = torch.from_numpy(
            (1.0 + self.sigma * rng.standard_normal(psf.shape)).astype(np.float32))
        psf = (psf * fac.clamp_min(0.0)).clamp_min(0.0)
        sample["psf"] = psf / psf.sum().clamp_min(1e-12)
        return sample


def restframe_shift(cube: torch.Tensor, v_star: torch.Tensor,
                    c_v: torch.Tensor, dloglam: float = 1e-4,
                    c_min: float = 0.5) -> torch.Tensor:
    """Des-desplaza cada espectro a rest-frame usando v_star (km/s).

    La rejilla MaNGA/MaNGIA es log10-lineal (dloglam = 1e-4 dex): un Doppler
    (1+v/c) es un desplazamiento CONSTANTE de log10(1+v/c)/dloglam canales,
    aplicado por spaxel con interpolacion lineal. Donde c_v <= c_min no se
    corrige (v poco confiable; specs/10 'Rest-frame opcional').

    cube: (L, H, W); v_star: (H, W) en km/s; c_v: (H, W) = c_phys[0].
    """
    L, H, W = cube.shape
    shift = torch.log10(1.0 + v_star / C_LIGHT_KMS) / dloglam
    shift = torch.where(c_v > c_min, shift, torch.zeros_like(shift))

    idx = (torch.arange(L, dtype=torch.float32).view(L, 1, 1)
           + shift.unsqueeze(0)).clamp(0.0, L - 1)
    lo = idx.floor().long()
    hi = (lo + 1).clamp_max(L - 1)
    frac = idx - lo.float()

    flat = cube.reshape(L, -1)
    cols = torch.arange(H * W).unsqueeze(0).expand(L, -1)
    out = (flat[lo.reshape(L, -1), cols] * (1.0 - frac.reshape(L, -1))
           + flat[hi.reshape(L, -1), cols] * frac.reshape(L, -1))
    return out.reshape(L, H, W)


class Compose:
    def __init__(self, transforms: list):
        self.transforms = list(transforms)

    def __call__(self, sample: dict, rng: np.random.Generator) -> dict:
        for t in self.transforms:
            sample = t(sample, rng)
        return sample
