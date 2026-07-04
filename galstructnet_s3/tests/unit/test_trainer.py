"""Tests de specs/60_training.md (Hito 5): 1 epoca, resume, Etapa 1
depende del mask_ratio (C6), particion GZ3D con assert."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("h5py")

from torch.utils.data import DataLoader

from galstructnet_s3.data import GalStructDataset, collate_pad
from galstructnet_s3.data.dataset import check_gz3d_partition
from galstructnet_s3.losses.total import GalStructNetLossV3
from galstructnet_s3.models.encoders.spectral import DilatedConv1DEncoder
from galstructnet_s3.models.model import build_model
from galstructnet_s3.training.curriculum import SpectralMAEPretrainer
from galstructnet_s3.training.trainer import Stage1Trainer, TrainerV3

CFG = {"model": {"spatial": "trivial", "physical": "std", "fusion": "concat",
                 "decoder": "trivial", "head": "evidence",
                 "spectral": {"backbone": "trivial"}},
       "loss": {"kappa": 0.25, "weights": {}},
       "training": {"stage": 2, "epochs": 1, "lr": 1e-3, "clip_norm": 1.0,
                    "precision": "fp32", "seed": 0}}


@pytest.fixture()
def loaders(synthetic_root):
    ds = GalStructDataset(synthetic_root, "train", mult=8)
    dl = DataLoader(ds, batch_size=1, collate_fn=collate_pad)
    return dl, dl


def _trainer(loaders, tmp_path, cfg=None):
    cfg = cfg or CFG
    model = build_model(cfg)
    loss_fn = GalStructNetLossV3(w_boundary=0.0, w_psf=0.0,
                                 kappa=cfg["loss"]["kappa"])
    return TrainerV3(model, loss_fn, cfg, *loaders, device="cpu",
                     ckpt_dir=tmp_path / "ckpt")


def test_01_una_epoca_sin_error(loaders, tmp_path):
    tr = _trainer(loaders, tmp_path)
    metrics = tr.run()
    assert "train/total" in metrics and "val/total" in metrics
    assert "val/rho_S_neff" in metrics            # la innovacion, logueada
    assert (tmp_path / "ckpt" / "last.pt").exists()


def test_02_resume_desde_checkpoint(loaders, tmp_path):
    tr = _trainer(loaders, tmp_path)
    tr.run()
    ck = tmp_path / "ckpt" / "last.pt"
    tr2 = _trainer(loaders, tmp_path)
    tr2.load_checkpoint(ck)
    assert tr2.epoch == 1
    p0 = next(iter(tr.model.state_dict().values()))
    p1 = next(iter(tr2.model.state_dict().values()))
    assert torch.equal(p0, p1)
    tr2.run()                                     # continua sin error
    assert tr2.epoch == 2


def test_03_etapa1_depende_del_mask_ratio(loaders, tmp_path):
    """C6: con mask_ratio=1.0 la loss es alta y baja al reducir el ratio
    (la tarea depende del enmascarado - lo que v2 no cumplia)."""
    dl, _ = loaders
    batch = next(iter(dl))
    losses = {}
    for ratio in (0.05, 1.0):
        torch.manual_seed(0)
        enc = DilatedConv1DEncoder(d_model=32, d_out=48, n_layers=2,
                                   return_sequence=True)
        pre = SpectralMAEPretrainer(enc, mask_ratio=ratio).eval()
        with torch.no_grad():
            out = pre(batch["cube"], batch["M"])
        losses[ratio] = float(out["loss"])
    assert losses[1.0] > losses[0.05]


def test_03b_etapa1_entrena(loaders, tmp_path):
    torch.manual_seed(0)
    enc = DilatedConv1DEncoder(d_model=32, d_out=48, n_layers=2,
                               return_sequence=True)
    pre = SpectralMAEPretrainer(enc, mask_ratio=0.3)
    cfg = {"training": {"epochs": 2, "lr": 1e-3, "clip_norm": 1.0}}
    tr = Stage1Trainer(pre, cfg, *loaders, device="cpu",
                       ckpt_dir=tmp_path / "s1")
    m = tr.run()
    assert "val/mae" in m and torch.isfinite(torch.tensor(m["val/mae"]))
    assert (tmp_path / "s1" / "stage1_last.pt").exists()


def test_04_particion_gz3d_disjunta(tmp_path):
    sp = tmp_path / "splits"
    sp.mkdir()
    (sp / "manga_gz3d_weak.txt").write_text("g1\ng2\n")
    (sp / "manga_gz3d_val.txt").write_text("g3\ng4\n")
    weak, val = check_gz3d_partition(tmp_path)
    assert weak == {"g1", "g2"} and val == {"g3", "g4"}
    (sp / "manga_gz3d_val.txt").write_text("g2\ng3\n")   # solapa
    with pytest.raises(AssertionError, match="particion GZ3D"):
        check_gz3d_partition(tmp_path)
