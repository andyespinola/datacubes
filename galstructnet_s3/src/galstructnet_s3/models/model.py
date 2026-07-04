"""Contenedor GalStructNetModel + modulos triviales del skeleton.

Spec: specs/00_ROADMAP.md Hito 2 - "skeleton end-to-end con modulos
triviales". Este modulo fija las INTERFACES entre componentes (contratos de
specs/20-45); los modulos reales de los Hitos 3-4 sustituyen a los triviales
via las banderas de config (`model.{spectral,spatial,physical,fusion,
decoder,head}`), sin tocar el contenedor.

Contrato de outputs del forward (consumido por GalStructNetLossV3, specs/50):

    {
      "lum":  {alpha, prob, evidence, vacuity},   # (B,5,H,W) / vacuity (B,1,H,W)
      "mass": {alpha, prob, evidence, vacuity},
      "alpha_obs_lum", "alpha_obs_mass",          # plano observado (PSF)
      "prob_obs_lum", "prob_obs_mass",
      "boundary": prob_lum,                        # entrada de boundary_loss
      "attn_w":  (B,3,H,W),                        # pesos por modalidad
      "c_fused": (B,1,H,W), "c_dec": (B,1,H,W),
    }
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .heads.psf import PSFEvidenceModule

N_CLASSES = 5


# --------------------------------------------------------------------------
# Modulos triviales (Hito 2). Sin pretension de calidad: solo interfaces.
# --------------------------------------------------------------------------

class TrivialSpectralEncoder(nn.Module):
    """cube (B,L,H,W) -> (B,d_out,H,W), agnostico a L: estadisticos por
    spaxel sobre lambda + mezcla 1x1. Interfaz de specs/20."""

    def __init__(self, d_out: int = 256):
        super().__init__()
        self.mix = nn.Conv2d(4, d_out, 1)

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        feats = torch.stack([cube.mean(1), cube.std(1),
                             cube.amin(1), cube.amax(1)], dim=1)
        return self.mix(feats)


class TrivialSpatialEncoder(nn.Module):
    """image (B,3,H,W) -> (features (B,d_out,H,W), skips x3 a d_out).
    Interfaz de specs/21; `mult` lo declara el encoder (trivial: 1)."""

    mult = 1

    def __init__(self, d_out: int = 256):
        super().__init__()
        self.proj = nn.Conv2d(3, d_out, 1)

    def forward(self, image: torch.Tensor):
        f = self.proj(image)
        skips = [F.avg_pool2d(f, 2 ** (i + 1), ceil_mode=True)
                 for i in range(3)]
        return f, skips


class TrivialPhysicalEncoder(nn.Module):
    """(maps, c_phys) (B,8,H,W) -> (features (B,64,H,W), c_out (B,1,H,W)).
    Interfaz de specs/22 (uniforme para Std/N/MP)."""

    def __init__(self, in_ch: int = 8, d_out: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, d_out, 1)

    def forward(self, maps: torch.Tensor, c_phys: torch.Tensor):
        return self.proj(maps * (c_phys > 0)), c_phys.mean(1, keepdim=True)


class TrivialFusion(nn.Module):
    """Concat + 1x1. Interfaz de specs/23: devuelve (F_fused, c_fused) y
    attn_w uniforme (las variantes reales lo aprenden)."""

    def __init__(self, d_spat=256, d_spec=256, d_phys=64, d=384):
        super().__init__()
        self.proj = nn.Conv2d(d_spat + d_spec + d_phys, d, 1)

    def forward(self, F_spat, F_spec, F_phys, cbar: dict):
        fused = self.proj(torch.cat([F_spat, F_spec, F_phys], dim=1))
        cstack = torch.cat([cbar["spat"], cbar["spec"], cbar["phys"]], dim=1)
        attn_w = torch.full_like(cstack, 1.0 / 3.0)
        c_fused = (attn_w * cstack).sum(1, keepdim=True)
        return fused, c_fused, attn_w


class TrivialDecoder(nn.Module):
    """(features (B,384,H,W), skips) -> (hidden (B,256,H,W), c_dec).
    Interfaz de specs/30 (c_fused pasa-traves como c_dec, variante std)."""

    def __init__(self, in_ch: int = 384, out_ch: int = 256):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, features, skips, c_fused, c_skips=None):
        hw = features.shape[-2:]
        x = self.proj(features)
        for s in skips:
            x = x + F.interpolate(s.mean(1, keepdim=True), size=hw,
                                  mode="bilinear", align_corners=False)
        return x, c_fused


class TrivialDualHeads(nn.Module):
    """hidden -> Dirichlet dual (mass/lum). Interfaz de specs/40:
    alpha >= 1, prob = alpha/S, evidence = alpha-1, vacuity = K/S."""

    def __init__(self, in_ch: int = 256, n_classes: int = N_CLASSES):
        super().__init__()
        self.proj = nn.ModuleDict({t: nn.Conv2d(in_ch, n_classes, 1)
                                   for t in ("mass", "lum")})

    def forward(self, hidden: torch.Tensor, c_dec: torch.Tensor) -> dict:
        out = {}
        for t, proj in self.proj.items():
            e = F.softplus(proj(hidden))
            alpha = e + 1.0
            S = alpha.sum(1, keepdim=True)
            out[t] = {"alpha": alpha, "prob": alpha / S, "evidence": e,
                      "vacuity": alpha.shape[1] / S}
        return out


# --------------------------------------------------------------------------
# Contenedor
# --------------------------------------------------------------------------

class GalStructNetModel(nn.Module):
    """Composicion encoders -> fusion -> decoder -> cabezas -> PSF.

    Los componentes llegan construidos (ver `build_model`); el contenedor
    solo define el flujo de datos y el dict de outputs (docstring del modulo).
    """

    def __init__(self, spectral: nn.Module, spatial: nn.Module,
                 physical: nn.Module, fusion: nn.Module, decoder: nn.Module,
                 heads: nn.Module, psf_mode: str = "evidence"):
        super().__init__()
        from .heads.psf import PSFProbModule
        self.spectral = spectral
        self.spatial = spatial
        self.physical = physical
        self.fusion = fusion
        self.decoder = decoder
        self.heads = heads
        self.psf_mode = psf_mode
        self.psf_module = (PSFEvidenceModule() if psf_mode == "evidence"
                           else PSFProbModule())

    @staticmethod
    def _c_skips(c_spat: torch.Tensor,
                 skips: list[torch.Tensor]) -> list[torch.Tensor]:
        """c de las skips Swin = avg-pool de c_spat a cada escala
        (aproximacion documentada, specs/45 'Integracion')."""
        c = c_spat.mean(1, keepdim=True)
        return [F.adaptive_avg_pool2d(c, (int(s.shape[-2]), int(s.shape[-1])))
                for s in skips]

    def forward(self, batch: dict) -> dict:
        F_spec = self.spectral(batch["cube"])
        F_spat, skips = self.spatial(batch["image"])
        F_phys, c_phys_out = self.physical(batch["maps"], batch["c_phys"])

        cbar = {"spat": batch["c_spat"].mean(1, keepdim=True),
                "spec": batch["c_spec"],
                "phys": c_phys_out}
        fused, c_fused, attn_w = self.fusion(F_spat, F_spec, F_phys, cbar)
        hidden, c_dec = self.decoder(fused, skips, c_fused,
                                     self._c_skips(batch["c_spat"], skips))

        out: dict = self.heads(hidden, c_dec)
        out["hidden"] = hidden                 # embeddings (MMD, cache)
        out["boundary"] = out["lum"]["prob"]
        out["attn_w"] = attn_w
        out["c_fused"] = c_fused
        out["c_dec"] = c_dec

        for t in ("mass", "lum"):
            src = (out[t]["alpha"] if self.psf_mode == "evidence"
                   else out[t]["prob"])
            alpha_obs, prob_obs = self.psf_module(src, batch["psf"])
            if alpha_obs is not None:
                out[f"alpha_obs_{t}"] = alpha_obs
            out[f"prob_obs_{t}"] = prob_obs
        return out


_NOT_YET = "componente '{key}: {val}' pendiente ({hito})"


def build_model(cfg: dict) -> GalStructNetModel:
    """Construye el modelo desde cfg['model'] (claves de configs/base.yaml:
    physical/fusion/decoder/head + spectral.backbone; las banderas son la
    escalera A0-A4 de specs/45).

    'trivial' (default, Hito 2) sigue disponible en todas las claves para
    tests de interfaz baratos.
    """
    from .decoder import FPNDecoder, FPNDecoderN
    from .encoders.physical import PhysicalEncoderN, PhysicalEncoderStd
    from .encoders.spatial import SwinSpatialEncoder
    from .encoders.spectral import build_spectral_encoder
    from .fusion_global import ConcatFusion, CrossAttentionFusion
    from .fusion_precision import PrecisionGatedFusion
    from .heads.segmentation import DualSegHeads

    m = cfg.get("model", {})

    def pick(key: str, registry: dict, default: str, hito: str):
        val = m.get(key, default)
        if val not in registry:
            raise NotImplementedError(_NOT_YET.format(key=key, val=val,
                                                      hito=hito))
        return registry[val]()

    spec_cfg = dict(m.get("spectral", {}))
    backbone = spec_cfg.pop("backbone", "trivial")
    if backbone == "trivial":
        spectral: nn.Module = TrivialSpectralEncoder()
    else:
        spectral = build_spectral_encoder(backbone, **spec_cfg)

    spatial = pick("spatial", {"trivial": TrivialSpatialEncoder,
                               "std": SwinSpatialEncoder},
                   "std", "specs/21")
    physical = pick("physical", {"trivial": TrivialPhysicalEncoder,
                                 "std": PhysicalEncoderStd,
                                 "normconv": PhysicalEncoderN},
                    "std", "specs/22 (moments: experimento lateral)")
    fusion = pick("fusion", {"trivial": TrivialFusion,
                             "concat": ConcatFusion,
                             "global": CrossAttentionFusion,
                             "precision": PrecisionGatedFusion},
                  "global", "specs/23")
    decoder = pick("decoder", {"trivial": TrivialDecoder,
                               "std": FPNDecoder,
                               "normconv": FPNDecoderN},
                   "std", "specs/30")
    share_gate = m.get("share_gate", True)
    b_init = m.get("b_init", 4.0)          # calibrar: init_evidence_scale.py
    heads = pick("head", {"trivial": TrivialDualHeads,
                          "std": lambda: DualSegHeads(kind="std"),
                          "evidence": lambda: DualSegHeads(
                              kind="evidence", share_gate=share_gate,
                              b_init=b_init)},
                 "std", "specs/40")
    return GalStructNetModel(spectral, spatial, physical, fusion, decoder,
                             heads, psf_mode=m.get("psf_mode", "evidence"))
