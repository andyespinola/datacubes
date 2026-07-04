"""Tests Hito 6: MMD (Etapa 4), consistencia D4 (Etapa 3), L_weak GZ3D."""
import pytest

torch = pytest.importorskip("torch")

from galstructnet_s3.losses.domain import mmd_loss, pooled_embeddings
from galstructnet_s3.training.curriculum import (consistency_loss,
                                                 weak_gz3d_loss)


def test_mmd_cero_misma_distribucion_positivo_distinta():
    torch.manual_seed(0)
    a = torch.randn(64, 16)
    b = torch.randn(64, 16)
    c = torch.randn(64, 16) + 3.0
    same = mmd_loss(a, b).item()
    diff = mmd_loss(a, c).item()
    assert same < 0.05
    assert diff > 10 * max(same, 1e-6)
    assert mmd_loss(a, a).item() < 1e-6 + same


def test_mmd_gradiente_y_vacio():
    a = torch.randn(8, 4, requires_grad=True)
    b = torch.randn(8, 4)
    mmd_loss(a, b).backward()
    assert torch.isfinite(a.grad).all()
    assert mmd_loss(torch.empty(0, 4), b).item() == 0.0


def test_pooled_embeddings_respeta_mascara():
    h = torch.ones(2, 3, 4, 4)
    h[:, :, 2:, :] = 100.0
    m = torch.zeros(2, 4, 4, dtype=torch.bool)
    m[:, :2, :] = True                      # solo la mitad de arriba
    emb = pooled_embeddings(h, m)
    assert torch.allclose(emb, torch.ones(2, 3))


def test_consistency_loss_cero_si_iguales_positiva_si_no():
    torch.manual_seed(0)
    p = torch.softmax(torch.randn(1, 5, 6, 6), dim=1)
    q = torch.softmax(torch.randn(1, 5, 6, 6), dim=1)
    m = torch.ones(1, 6, 6, dtype=torch.bool)
    assert consistency_loss(p, p, m).item() < 1e-8
    assert consistency_loss(p, q, m).item() > 0


def test_weak_gz3d_solo_bar_arm():
    torch.manual_seed(0)
    B, H, W = 1, 6, 6
    prob = torch.softmax(torch.randn(B, 5, H, W, requires_grad=False), dim=1)
    prob.requires_grad_(True)
    out = {"prob_obs_lum": prob}
    frac = torch.rand(B, 2, H, W)
    cover = torch.zeros(B, 2, H, W, dtype=torch.bool)
    cover[:, :, :3, :] = True
    batch = {"gz3d_frac": frac, "gz3d_mask": cover}
    loss = weak_gz3d_loss(out, batch)
    loss.backward()
    g = prob.grad
    # gradiente solo en canales bar(2)/arm(3) y solo bajo cobertura
    assert g[:, [0, 1, 4]].abs().sum() == 0
    assert g[:, 2, :3].abs().sum() > 0
    assert g[:, 2, 3:].abs().sum() == 0


def test_weak_gz3d_minimo_en_target():
    B, H, W = 1, 4, 4
    frac = torch.rand(B, 2, H, W) * 0.8 + 0.1
    prob = torch.zeros(B, 5, H, W)
    prob[:, 2] = frac[:, 0]
    prob[:, 3] = frac[:, 1]
    batch = {"gz3d_frac": frac,
             "gz3d_mask": torch.ones(B, 2, H, W, dtype=torch.bool)}
    at_target = weak_gz3d_loss({"prob_obs_lum": prob}, batch).item()
    prob2 = prob.clone()
    prob2[:, 2] = (frac[:, 0] + 0.3).clamp_max(0.99)
    off = weak_gz3d_loss({"prob_obs_lum": prob2}, batch).item()
    assert at_target < off
