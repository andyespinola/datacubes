"""Hito 3: baseline A0 real end-to-end sobre la fixture.

Encoder espectral en variante conv1d (A8) para CI sin CUDA; el resto es el
A0 de specs (Swin+FPN, fisico Std, fusion global, FPNDecoder, cabeza std).
Incluye el test de overfitting a 1 muestra (specs/00 regla 3, specs/30).
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("h5py")
pytest.importorskip("timm")

from torch.utils.data import DataLoader

from galstructnet_s3.data import GalStructDataset, collate_pad
from galstructnet_s3.losses.dirichlet import anchored_seg_loss
from galstructnet_s3.models.model import build_model

A0_CFG = {"model": {
    "spatial": "std", "physical": "std", "fusion": "global",
    "decoder": "std", "head": "std", "psf_mode": "evidence",
    "spectral": {"backbone": "conv1d", "d_model": 32, "d_out": 256,
                 "n_layers": 2},
}}

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _to(batch, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@pytest.fixture()
def batch(synthetic_root):
    ds = GalStructDataset(synthetic_root, "train", mult=32)
    return _to(next(iter(DataLoader(ds, batch_size=1,
                                    collate_fn=collate_pad))), DEV)


def test_a0_forward_shapes_y_backward(batch):
    model = build_model(A0_CFG).to(DEV)
    out = model(batch)
    B, _, H, W = batch["cube"].shape
    for t in ("mass", "lum"):
        assert out[t]["alpha"].shape == (B, 5, H, W)
        assert (out[t]["alpha"] >= 1).all()
        assert out[f"alpha_obs_{t}"].shape == (B, 5, H, W)
    ms = batch["M"] & ~batch["M_unc_lum"]
    loss = anchored_seg_loss(out["lum"]["alpha"], batch["Y_lum"],
                             batch["n_eff_lum"], ms, kappa=0.5,
                             n_eff_cap=None)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.fixture()
def batch_rings(synthetic_root, tmp_path):
    """Entry con etiquetas APRENDIBLES (5 anillos radiales). Las etiquetas
    de la fixture base son ruido blanco por spaxel — imposibles de
    overfittear con features suaves; el criterio de overfitting de los
    specs asume el piloto real, donde la etiqueta correlaciona con la
    senal."""
    import shutil

    import h5py
    import numpy as np

    name = "dataset_entry_SYNTH-0000_v0.h5"
    root = tmp_path / "entries_rings"
    root.mkdir()
    shutil.copy(synthetic_root / name, root / name)
    with h5py.File(root / name, "r+") as f:
        H, W = f["masks/M_valid"].shape
        yy, xx = np.mgrid[0:H, 0:W]
        cls = np.digitize(np.hypot(yy - H / 2, xx - W / 2), [3, 7, 11, 15])
        Y = np.full((5, H, W), 0.02, np.float32)
        for k in range(5):
            Y[k][cls == k] = 0.92
        Y /= Y.sum(0)
        for w in ("mass", "lum"):
            f[f"labels/Y_{w}_raw"][...] = Y
            f[f"labels/Y_{w}_psf"][...] = Y
    ds = GalStructDataset(root, "train", mult=32)
    return _to(next(iter(DataLoader(ds, batch_size=1,
                                    collate_fn=collate_pad))), DEV)


def test_a0_overfitting_1_muestra(batch_rings):
    batch = batch_rings
    torch.manual_seed(0)
    model = build_model(A0_CFG).to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    ms = batch["M"] & ~batch["M_unc_lum"]

    def step():
        opt.zero_grad()
        out = model(batch)
        loss = anchored_seg_loss(out["lum"]["alpha"], batch["Y_lum"],
                                 batch["n_eff_lum"], ms, kappa=0.5,
                                 n_eff_cap=None)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return loss.item(), out

    first, _ = step()
    for _ in range(149):
        last, out = step()
    assert last < first * 0.5, (first, last)      # baja >50% en 150 iter
    acc = (out["lum"]["prob"].argmax(1)
           == batch["Y_lum"].argmax(1))[ms].float().mean().item()
    assert acc > 0.7, acc
