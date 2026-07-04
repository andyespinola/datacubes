"""Tests de specs/43_module_psf.md - modo evidence EJECUTABLE."""
import pytest

torch = pytest.importorskip("torch")
from galstructnet_s3.models.heads.psf import PSFEvidenceModule

mod = PSFEvidenceModule()


def _delta_kernel(B=1, k=5):
    ker = torch.zeros(B, k, k)
    ker[:, k // 2, k // 2] = 1.0
    return ker


def test_02_suma_a_uno():
    torch.manual_seed(0)
    a = 1.0 + torch.rand(2, 5, 8, 8) * 10
    ker = torch.rand(2, 5, 5)
    ker = ker / ker.sum((-2, -1), keepdim=True)
    _, prob = mod(a, ker)
    assert torch.allclose(prob.sum(1), torch.ones_like(prob.sum(1)), atol=1e-6)


def test_03_identidad_con_delta():
    torch.manual_seed(1)
    a = 1.0 + torch.rand(1, 5, 8, 8) * 10
    a_obs, _ = mod(a, _delta_kernel())
    assert torch.allclose(a_obs, a, atol=1e-5)


def test_04_conservacion_interior():
    a = torch.ones(1, 5, 17, 17)
    a[0, 2, 8, 8] = 101.0                                      # delta central
    ker = torch.full((1, 5, 5), 1.0 / 25)
    a_obs, _ = mod(a, ker)
    e_in = (a - 1).sum().item()
    e_out = (a_obs - 1).sum().item()
    assert abs(e_in - e_out) < 1e-4


def test_06_gradiente_fluye():
    a = (1.0 + torch.rand(1, 5, 6, 6)).requires_grad_(True)
    ker = torch.full((1, 3, 3), 1.0 / 9)
    a_obs, prob = mod(a, ker)
    prob.sum().backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
