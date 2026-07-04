"""Curvas de robustez (specs/45 experimentos firma) y compare_with_gz3d
(specs/70) sobre la fixture."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("h5py")

from torch.utils.data import DataLoader

from galstructnet_s3.data import GalStructDataset, collate_pad
from galstructnet_s3.evaluation.validation import (compare_with_gz3d,
                                                   degrade_batch_dropout,
                                                   degrade_batch_noise,
                                                   robustness_sweep)
from galstructnet_s3.models.model import build_model

CFG = {"model": {"spatial": "trivial", "physical": "normconv",
                 "fusion": "precision", "decoder": "trivial",
                 "head": "evidence", "spectral": {"backbone": "trivial"}}}


@pytest.fixture()
def batch(synthetic_root):
    ds = GalStructDataset(synthetic_root, "train", mult=8)
    return next(iter(DataLoader(ds, batch_size=1, collate_fn=collate_pad)))


def test_degrade_noise_actualiza_c(batch):
    torch.manual_seed(0)
    out = degrade_batch_noise(batch, factor=4.0)
    M = batch["M"]
    assert (out["c_phys"][..., M.squeeze(0)]
            < batch["c_phys"][..., M.squeeze(0)]).all()
    assert not torch.equal(out["maps"], batch["maps"])
    # factor 1 = identidad
    same = degrade_batch_noise(batch, factor=1.0)
    assert torch.equal(same["maps"], batch["maps"])


def test_degrade_dropout_apaga_c(batch):
    g = torch.Generator().manual_seed(0)
    out = degrade_batch_dropout(batch, frac=0.3, rng=g)
    d = out["dropped"]
    assert d.any()
    assert (out["c_phys"][..., d.squeeze(0)] == 0).all()
    assert (out["c_spec"][..., d.squeeze(0)] == 0).all()
    frac = float(d.sum()) / float(batch["M"].sum())
    assert 0.15 < frac < 0.45


def test_robustness_sweep_produce_curva(batch):
    torch.manual_seed(0)
    model = build_model(CFG)
    curve = robustness_sweep(model, batch, degrade_batch_noise,
                             levels=(1.0, 4.0))
    assert set(curve) == {1.0, 4.0}
    for lvl in curve.values():
        assert "soft_nll" in lvl and "iou_med" in lvl


def test_compare_with_gz3d_particionado(batch, tmp_path):
    sp = tmp_path / "splits"
    sp.mkdir()
    (sp / "manga_gz3d_weak.txt").write_text("g1\n")
    (sp / "manga_gz3d_val.txt").write_text("SYNTH-0000\n")
    torch.manual_seed(0)
    model = build_model(CFG)
    B, H, W = batch["M"].shape
    batch["gz3d_frac"] = torch.rand(B, 2, H, W)
    batch["gz3d_mask"] = batch["M"].unsqueeze(1).expand(B, 2, H, W).clone()
    res = compare_with_gz3d(model, [batch], tmp_path)
    assert set(res) == {"iou_bar", "iou_arm", "vote_corr"}
    # particion invalida => assert
    (sp / "manga_gz3d_val.txt").write_text("g1\n")
    with pytest.raises(AssertionError, match="particion GZ3D"):
        compare_with_gz3d(model, [batch], tmp_path)
