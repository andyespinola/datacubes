"""Tests 1-6 de specs/42_head_uncertainty.md - EJECUTABLES (referencia impl)."""
import math
import pytest

torch = pytest.importorskip("torch")
from galstructnet_s3.models.heads.uncertainty import UncertaintyDecomposition

K = 5
dec = UncertaintyDecomposition(K)


def _alpha(vals):
    return torch.tensor(vals, dtype=torch.float64).view(1, K, 1, 1)


def test_02_caso_ignorante_valores_exactos():
    out = dec(_alpha([1, 1, 1, 1, 1]))
    AU_exact = sum(1.0 / n for n in range(2, K + 1))          # psi(K+1)-psi(2)
    assert math.isclose(out["total"].item(), math.log(K), rel_tol=1e-6)
    assert math.isclose(out["aleatoric"].item(), AU_exact, rel_tol=1e-5)
    assert out["epistemic"].item() > 0.3
    assert math.isclose(out["vacuity"].item(), 1.0, rel_tol=1e-6)


def test_03_caso_concentrado():
    out = dec(_alpha([1000, 1, 1, 1, 1]))
    assert out["total"].item() < 0.05
    assert out["epistemic"].item() < 0.05
    assert out["vacuity"].item() < 0.01


def test_04_mezcla_con_mucha_evidencia():
    out = dec(_alpha([500, 500, 1, 1, 1]))
    assert abs(out["aleatoric"].item() - math.log(2)) < 0.05   # AU alta
    assert out["epistemic"].item() < 0.02                       # EU ~ 0


def test_05_jensen_au_le_tu():
    torch.manual_seed(0)
    a = 1.0 + torch.rand(4, K, 7, 7).double() * 50
    out = dec(a)
    assert (out["aleatoric"] <= out["total"] + 1e-6).all()
    assert (out["epistemic"] >= 0).all()


def test_01_rangos():
    torch.manual_seed(1)
    a = 1.0 + torch.rand(2, K, 5, 5).double() * 20
    out = dec(a)
    assert (out["total"] <= math.log(K) + 1e-6).all()
    assert (out["vacuity"] > 0).all() and (out["vacuity"] <= 1.0 + 1e-6).all()
