"""Padding dinamico y collate. Spec: specs/10_dataset.md - Hito 1."""
from __future__ import annotations
import torch
import torch.nn.functional as F

_SPATIAL_KEYS_2D = ("M", "M_unc_mass", "M_unc_lum", "w_phys_mass",
                    "n_eff_mass", "n_eff_mass_obs", "n_eff_lum", "n_eff_lum_obs")


def pad_to_multiple(sample: dict, mult: int = 32) -> dict:
    """Pad de TODOS los tensores espaciales al multiplo `mult`.

    Certeza y M_valid se rellenan con 0/False: el modelo ignora el padding
    por construccion (specs/45 P1). Guarda 'hw_native' para recorte en eval.
    """
    ref = sample["image"]
    H, W = ref.shape[-2:]
    ph, pw = (-H) % mult, (-W) % mult
    if ph == 0 and pw == 0:
        sample["hw_native"] = (H, W)
        return sample
    pad = (0, pw, 0, ph)  # left,right,top,bottom -> (W_l,W_r,H_t,H_b) via F.pad last two dims
    for k, v in sample.items():
        if not torch.is_tensor(v):
            continue
        if v.dim() >= 2 and v.shape[-2:] == (H, W):
            if v.dtype == torch.bool:
                sample[k] = F.pad(v, pad, value=False)
            else:
                sample[k] = F.pad(v, pad, value=0.0)
    sample["hw_native"] = (H, W)
    return sample


def collate_pad(batch: list[dict]) -> dict:
    """Pad por batch a (H_max, W_max) del batch (ya multiplos de `mult`) y
    stack. Devuelve 'hw_native' por sample para recortar en evaluacion.

    - Tensores espaciales (shape[-2:] == (H_i, W_i) del sample): pad con
      0/False y stack.
    - PSF: los kernels pueden variar de K entre bundles; se paddean al K_max
      del batch (kernel centrado, K impar) y se stackean.
    - Escalares/vectores (fracciones): stack directo. Strings/ints: lista.
    """
    out: dict = {}
    hw = [s["hw_native"] for s in batch]
    ref_hw = [tuple(s["M"].shape[-2:]) for s in batch]  # tamano actual (padded)
    H_max = max(h for h, _ in ref_hw)
    W_max = max(w for _, w in ref_hw)

    for k in batch[0]:
        if k == "hw_native":
            out[k] = hw
            continue
        v0 = batch[0][k]
        if not torch.is_tensor(v0):
            out[k] = [s[k] for s in batch]
            continue
        if k == "psf":
            K_max = max(s[k].shape[-1] for s in batch)
            padded = []
            for s in batch:
                p = s[k]
                d = (K_max - p.shape[-1]) // 2
                padded.append(F.pad(p, (d, d, d, d)) if d else p)
            out[k] = torch.stack(padded)
        elif v0.dim() >= 2 and tuple(v0.shape[-2:]) == ref_hw[0]:
            padded = []
            for s, (h, w) in zip(batch, ref_hw):
                v = s[k]
                pad = (0, W_max - w, 0, H_max - h)
                if pad[1] or pad[3]:
                    v = F.pad(v, pad, value=False if v.dtype == torch.bool
                              else 0.0)
                padded.append(v)
            out[k] = torch.stack(padded)
        else:
            out[k] = torch.stack([s[k] for s in batch])
    return out
