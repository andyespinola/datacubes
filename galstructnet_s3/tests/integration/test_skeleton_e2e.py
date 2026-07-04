"""Hito 2: skeleton end-to-end trivial sobre la fixture (specs/00 Hito 2).

dataset -> collate -> modelo trivial -> perdidas ya implementadas -> backward.
El piloto real (TNG50-87-141934-0-127) sustituye a la fixture fuera de CI.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("h5py")

from torch.utils.data import DataLoader

from galstructnet_s3.data import GalStructDataset, collate_pad
from galstructnet_s3.losses.dirichlet import anchored_seg_loss
from galstructnet_s3.losses.physics import physics_constraint_loss
from galstructnet_s3.models.heads.boundary import boundary_loss
from galstructnet_s3.models.model import build_model


@pytest.fixture()
def batch(synthetic_entry):
    ds = GalStructDataset(synthetic_entry.parent, "train", mult=8)
    dl = DataLoader(ds, batch_size=1, collate_fn=collate_pad)
    return next(iter(dl))


TRIVIAL_CFG = {"model": {"spatial": "trivial", "physical": "trivial",
                         "fusion": "trivial", "decoder": "trivial",
                         "head": "trivial",
                         "spectral": {"backbone": "trivial"}}}


@pytest.fixture()
def model():
    return build_model(TRIVIAL_CFG)  # todo trivial (skeleton Hito 2)


def test_forward_shapes_e_invariantes(batch, model):
    out = model(batch)
    B, _, H, W = batch["cube"].shape
    for t in ("mass", "lum"):
        assert out[t]["alpha"].shape == (B, 5, H, W)
        assert (out[t]["alpha"] >= 1).all()
        s = out[t]["prob"].sum(1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-6)
        assert out[t]["vacuity"].shape == (B, 1, H, W)
        assert out[f"alpha_obs_{t}"].shape == (B, 5, H, W)
    assert out["attn_w"].shape == (B, 3, H, W)
    assert out["c_fused"].shape == (B, 1, H, W)
    assert out["boundary"].shape == (B, 5, H, W)


def test_backward_finito_con_perdidas_reales(batch, model):
    out = model(batch)
    ms = batch["M"] & ~batch["M_unc_lum"]
    loss = anchored_seg_loss(out["lum"]["alpha"], batch["Y_lum"],
                             batch["n_eff_lum"], ms, kappa=0.5,
                             n_eff_cap=None)
    loss = loss + anchored_seg_loss(out["mass"]["alpha"], batch["Y_mass"],
                                    batch["n_eff_mass"],
                                    batch["M"] & ~batch["M_unc_mass"],
                                    kappa=0.5, n_eff_cap=None)
    loss = loss + boundary_loss(out["boundary"], batch["Y_lum"], batch["M"])
    loss = loss + physics_constraint_loss(
        out["lum"]["prob"], batch["M"], batch["w_phys_mass"],
        batch["target_fractions_mass"])
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), name


def test_config_no_implementada_falla_claro():
    with pytest.raises(NotImplementedError, match="specs/22"):
        build_model({"model": {"physical": "moments",
                               "spectral": {"backbone": "trivial"}}})


def test_size_agnostic(batch, model):
    # el mismo modelo forward con otro tamano espacial (sin literales)
    out_a = model(batch)
    crop = {k: (v[..., :16, :24] if torch.is_tensor(v) and v.dim() >= 3
                and v.shape[-2:] == batch["M"].shape[-2:] else v)
            for k, v in batch.items()}
    crop["M"] = batch["M"][..., :16, :24]
    out_b = model(crop)
    assert out_b["lum"]["alpha"].shape[-2:] == (16, 24)
    assert out_a["lum"]["alpha"].shape[-2:] == batch["M"].shape[-2:]
