"""La escalera A0-A4 es construible desde configs (specs/45 criterio de
aceptacion; specs/60 'Escalera'). Cada YAML produce un modelo que forwardea
la fixture con shapes correctos."""
import glob

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("h5py")
pytest.importorskip("timm")
pytest.importorskip("yaml")

from torch.utils.data import DataLoader

from galstructnet_s3.config import load_config
from galstructnet_s3.data import GalStructDataset, collate_pad
from galstructnet_s3.models.model import build_model

CONFIGS = sorted(glob.glob("configs/ablation_epn/*.yaml"))


def test_hay_configs_de_escalera():
    assert len(CONFIGS) >= 6      # A0, A1, A2a-c, A3, A4x2


@pytest.mark.parametrize("path", CONFIGS)
def test_config_construye_y_forwardea(path, synthetic_root):
    cfg = load_config(path)
    # CI: el backbone mamba solo con CUDA; conv1d es el contrafactual A8
    if not torch.cuda.is_available():
        cfg["model"]["spectral"]["backbone"] = "conv1d"
    cfg["model"]["spectral"].setdefault("d_model", 32)
    cfg["model"]["spectral"]["n_layers"] = 2

    model = build_model(cfg)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()

    ds = GalStructDataset(synthetic_root, "train", mult=32)
    batch = next(iter(DataLoader(ds, batch_size=1, collate_fn=collate_pad)))
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
             for k, v in batch.items()}
    with torch.no_grad():
        out = model(batch)
    B, _, H, W = batch["cube"].shape
    for t in ("mass", "lum"):
        assert out[t]["alpha"].shape == (B, 5, H, W)
        assert (out[t]["alpha"] >= 1).all()
        assert out[f"prob_obs_{t}"].shape == (B, 5, H, W)
    if cfg["model"].get("psf_mode", "evidence") == "evidence":
        assert "alpha_obs_lum" in out
    else:
        assert "alpha_obs_lum" not in out


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="bf16 requiere GPU Ampere+ (la maquina de entrenamiento)")
def test_a3_bf16_forward_backward(synthetic_root):
    """Criterio specs/45: forward P1-P4 con gradientes finitos en bf16."""
    from galstructnet_s3.losses.total import GalStructNetLossV3

    cfg = load_config("configs/ablation_epn/A3_evidence_head.yaml")
    cfg["model"]["spectral"].update({"d_model": 32, "n_layers": 2})
    model = build_model(cfg).cuda()
    ds = GalStructDataset(synthetic_root, "train", mult=32)
    batch = next(iter(DataLoader(ds, batch_size=1, collate_fn=collate_pad)))
    batch = {k: (v.cuda() if torch.is_tensor(v) else v)
             for k, v in batch.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(batch)
        L = GalStructNetLossV3(w_psf=0.0)(out, batch)  # KL en fp32 interno
    assert torch.isfinite(L["total"])
    L["total"].backward()
    for n, p in model.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), n
