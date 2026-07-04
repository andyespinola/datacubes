"""Tests ejecutables: boundary (specs/41), metricas suaves (specs/70),
to_certainty y pad_to_multiple (specs/10)."""
import pytest

torch = pytest.importorskip("torch")
from galstructnet_s3.models.heads.boundary import boundary_map, boundary_loss
from galstructnet_s3.evaluation.metrics import soft_brier, soft_nll
from galstructnet_s3.data.dataset import to_certainty
from galstructnet_s3.data.collate import pad_to_multiple


def test_boundary_constante_es_interior():
    prob = torch.zeros(1, 5, 8, 8)
    prob[:, 1] = 1.0
    B = boundary_map(prob)
    assert (B[:, :, 1:-1, 1:-1] > 0.999).all()


def test_boundary_loss_cero_si_iguales():
    torch.manual_seed(0)
    Y = torch.softmax(torch.randn(1, 5, 8, 8), dim=1)
    m = torch.ones(1, 8, 8, dtype=torch.bool)
    assert boundary_loss(Y, Y, m).item() < 1e-8


def test_soft_scores_minimos_en_target():
    torch.manual_seed(0)
    Y = torch.softmax(torch.randn(2, 5, 6, 6), dim=1)
    m = torch.ones(2, 6, 6, dtype=torch.bool)
    assert soft_brier(Y, Y, m).item() < 1e-8
    other = torch.softmax(torch.randn(2, 5, 6, 6), dim=1)
    assert soft_nll(Y, Y, m).item() <= soft_nll(other, Y, m).item()


def test_to_certainty_semantica():
    sref = torch.tensor(2.0)
    assert abs(to_certainty(sref.clone(), sref).item() - 0.5) < 1e-6
    assert to_certainty(torch.tensor(0.0), sref).item() == 1.0
    assert to_certainty(torch.tensor(1e6), sref).item() < 1e-6


def test_pad_to_multiple_contrato():
    s = {"image": torch.rand(3, 34, 35), "M": torch.ones(34, 35, dtype=torch.bool),
         "c_spat": torch.rand(3, 34, 35)}
    out = pad_to_multiple(s, mult=32)
    assert out["image"].shape[-2:] == (64, 64)
    assert out["hw_native"] == (34, 35)
    assert out["M"][40, 40].item() is False
    assert float(out["c_spat"][..., 40, 40].sum()) == 0.0
