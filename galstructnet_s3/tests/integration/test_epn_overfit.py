"""Test de overfitting EPN (specs/45 'Test de overfitting', Hito 3/4):
encoder fisico-N + EvidenceHead sobre 1 muestra; la KL anclada baja y
rho_Spearman(S, kappa*N_eff) se vuelve positiva.

La fixture base tiene etiquetas/errores de ruido blanco (no aprendibles);
aqui se construye una variante FISICAMENTE COHERENTE: clases por anillos
radiales, masa radial y error instrumental menor donde hay mas senal
(como pyPipe3D real) - la correlacion c <-> N_eff es el mecanismo que la
familia EPN explota. En el piloto real este test corre tal cual sobre el
dataset_entry del TNG50-87-141934-0-127.
"""
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")
scipy_stats = pytest.importorskip("scipy.stats")

from torch.utils.data import DataLoader

from galstructnet_s3.data import GalStructDataset, collate_pad
from galstructnet_s3.losses.dirichlet import anchored_seg_loss
from galstructnet_s3.models.encoders.physical import PhysicalEncoderN
from galstructnet_s3.models.heads.segmentation import EvidenceHead


@pytest.fixture()
def batch_coherente(synthetic_root, tmp_path):
    name = "dataset_entry_SYNTH-0000_v0.h5"
    root = tmp_path / "entries_epn"
    root.mkdir()
    shutil.copy(synthetic_root / name, root / name)
    with h5py.File(root / name, "r+") as f:
        H, W = f["masks/M_valid"].shape
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.hypot(yy - H / 2, xx - W / 2)
        cls = np.digitize(r, [3, 7, 11, 15])
        Y = np.full((5, H, W), 0.02, np.float32)
        for k in range(5):
            Y[k][cls == k] = 0.92
        Y /= Y.sum(0)
        maps = f["inputs/pipe3d_maps"][()]
        maps[4] = np.exp(-r / 6.0)                 # masa radial
        f["inputs/pipe3d_maps"][...] = maps
        err = (0.05 + 0.6 * (1 - np.exp(-r / 6.0))).astype(np.float32)
        f["inputs/pipe3d_err"][...] = np.repeat(err[None], 8, axis=0)
        for w in ("mass", "lum"):
            f[f"labels/Y_{w}_raw"][...] = Y
            f[f"labels/Y_{w}_psf"][...] = Y
    ds = GalStructDataset(root, "train")
    return next(iter(DataLoader(ds, batch_size=1, collate_fn=collate_pad)))


def test_overfit_encoder_n_mas_evidence_head(batch_coherente):
    import sys
    sys.path.insert(0, "scripts")
    from init_evidence_scale import solve_b

    b = batch_coherente
    ms = b["M"] & ~b["M_unc_lum"]
    neff = b["n_eff_lum"][ms]
    kappa = 0.25                                   # dial correcto del gap
    cap = float(np.percentile(neff.numpy(), 99))   # p99 (specs/50)
    b_init = solve_b(kappa, float(neff.median()))

    torch.manual_seed(0)
    enc = PhysicalEncoderN()
    head = EvidenceHead(in_ch=64, b_init=b_init)
    opt = torch.optim.AdamW([
        {"params": enc.parameters(), "lr": 1e-3},
        {"params": head.parameters(), "lr": 5e-3}])

    losses = []
    for _ in range(601):
        opt.zero_grad()
        feats, c_out = enc(b["maps"], b["c_phys"])
        out = head(feats, c_out)
        loss = anchored_seg_loss(out["alpha"], b["Y_lum"], b["n_eff_lum"],
                                 ms, kappa=kappa, n_eff_cap=cap)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # KL anclada baja (>90%, tendencia monotona salvo ruido de Adam)
    assert losses[-1] < 0.1 * losses[0], (losses[0], losses[-1])
    smooth = np.convolve(losses, np.ones(50) / 50, mode="valid")
    assert smooth[-1] < smooth[len(smooth) // 2] < smooth[0]

    # la concentracion trackea la estadistica fisica de la etiqueta
    S = out["alpha"].sum(1)[ms].detach().numpy()
    target = (kappa * b["n_eff_lum"].clamp_max(cap))[ms].numpy()
    rho = scipy_stats.spearmanr(S, target).statistic
    assert rho > 0.3, rho
