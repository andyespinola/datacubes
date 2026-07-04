"""Modulo PSF-aware. Spec: specs/43_module_psf.md - Hito 4.

Modo 'evidence' (default): alpha_obs = K_PSF * (alpha - 1) + 1.
IMPLEMENTACION DE REFERENCIA transcrita del spec; kernel por muestra via
grouped conv. Modo 'prob' (baseline A4): migrar codigo v2.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn


class PSFEvidenceModule(nn.Module):
    """La PSF redistribuye pseudo-conteos: coherente con S ~ kappa*N_eff."""

    def forward(self, alpha_int: torch.Tensor, psf_kernel: torch.Tensor):
        B, K, H, W = alpha_int.shape
        kh, kw = psf_kernel.shape[-2:]
        psf = psf_kernel / psf_kernel.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        e = (alpha_int - 1.0).reshape(1, B * K, H, W)
        w = psf.unsqueeze(1).expand(B, K, kh, kw).reshape(B * K, 1, kh, kw)
        e_obs = F.conv2d(e, w.to(e.dtype), padding=(kh // 2, kw // 2),
                         groups=B * K).reshape(B, K, H, W)
        alpha_obs = e_obs + 1.0
        return alpha_obs, alpha_obs / alpha_obs.sum(dim=1, keepdim=True)


class PSFProbModule(nn.Module):
    """Baseline A4 (modo 'prob', v2): conv del kernel sobre prob +
    renormalizacion. Interfaz espejo del modo evidencia: devuelve
    (None, prob_obs) - no existe alpha_obs en este modo."""

    def forward(self, prob_int: torch.Tensor, psf_kernel: torch.Tensor):
        B, K, H, W = prob_int.shape
        kh, kw = psf_kernel.shape[-2:]
        psf = psf_kernel / psf_kernel.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        p = prob_int.reshape(1, B * K, H, W)
        w = psf.unsqueeze(1).expand(B, K, kh, kw).reshape(B * K, 1, kh, kw)
        p_obs = F.conv2d(p, w.to(p.dtype), padding=(kh // 2, kw // 2),
                         groups=B * K).reshape(B, K, H, W)
        prob_obs = p_obs / p_obs.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return None, prob_obs
