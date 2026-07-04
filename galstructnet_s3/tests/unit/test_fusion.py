"""Tests de specs/23_fusion.md. concat/global (Hito 3); precision en Hito 4."""
import pytest

torch = pytest.importorskip("torch")
from galstructnet_s3.models.fusion_global import ConcatFusion, CrossAttentionFusion
from galstructnet_s3.models.fusion_precision import PrecisionGatedFusion

VARIANTS = [ConcatFusion, CrossAttentionFusion, PrecisionGatedFusion]


def _inputs(B=2, H=18, W=22, seed=0):
    torch.manual_seed(seed)
    F_spat = torch.randn(B, 256, H, W, requires_grad=True)
    F_spec = torch.randn(B, 256, H, W, requires_grad=True)
    F_phys = torch.randn(B, 64, H, W, requires_grad=True)
    cbar = {m: torch.rand(B, 1, H, W) for m in ("spat", "spec", "phys")}
    return F_spat, F_spec, F_phys, cbar


@pytest.mark.parametrize("cls", VARIANTS)
@pytest.mark.parametrize("B,H,W", [(1, 10, 14), (2, 18, 22)])
def test_01_shapes_hw_arbitrarios_b1(cls, B, H, W):
    F_spat, F_spec, F_phys, cbar = _inputs(B=B, H=H, W=W)
    fused, c_fused, attn_w = cls()(F_spat, F_spec, F_phys, cbar)
    assert fused.shape == (B, 384, H, W)
    assert c_fused.shape == (B, 1, H, W)
    assert attn_w is None or attn_w.shape == (B, 3, H, W)


@pytest.mark.parametrize("cls", VARIANTS)
def test_02_gradient_flow_tres_modalidades(cls):
    F_spat, F_spec, F_phys, cbar = _inputs()
    fused, _, _ = cls()(F_spat, F_spec, F_phys, cbar)
    fused.sum().backward()
    for name, t in (("spat", F_spat), ("spec", F_spec), ("phys", F_phys)):
        assert t.grad is not None and t.grad.abs().sum() > 0, name


@pytest.mark.parametrize("cls", VARIANTS)
def test_03_determinismo_eval(cls):
    F_spat, F_spec, F_phys, cbar = _inputs()
    mod = cls().eval()
    with torch.no_grad():
        a, _, _ = mod(F_spat, F_spec, F_phys, cbar)
        b, _, _ = mod(F_spat, F_spec, F_phys, cbar)
    assert torch.equal(a, b)


def test_04_degradacion_selectiva_attn():
    # cubierta en detalle por test_evidence_layers::test_08 (mismo criterio)
    torch.manual_seed(0)
    fus = PrecisionGatedFusion()
    F_spat, F_spec, F_phys, cbar = _inputs()
    opt = torch.optim.Adam(fus.parameters(), lr=1e-2)
    for _ in range(30):
        opt.zero_grad()
        _, _, w = fus(F_spat, F_spec, F_phys, cbar)
        ((w[:, 1] - cbar["spec"].squeeze(1)) ** 2).mean().backward()
        opt.step()
    fus.eval()
    low = {k: v.clone() for k, v in cbar.items()}
    low["spec"][:, :, :8, :] = 0.0
    with torch.no_grad():
        _, _, w_hi = fus(F_spat, F_spec, F_phys, cbar)
        _, _, w_lo = fus(F_spat, F_spec, F_phys, low)
    assert w_lo[:, 1, :8, :].mean() < w_hi[:, 1, :8, :].mean()


def test_05_neutralidad_inicial():
    F_spat, F_spec, F_phys, cbar = _inputs()
    torch.manual_seed(5)
    fus = PrecisionGatedFusion().eval()
    other = {m: torch.rand_like(c) for m, c in cbar.items()}
    with torch.no_grad():
        fa, _, _ = fus(F_spat, F_spec, F_phys, cbar)
        fb, _, _ = fus(F_spat, F_spec, F_phys, other)
    assert torch.allclose(fa, fb, atol=1e-6)   # g_m init->0: c no influye


def test_06_convexidad_attn_w():
    F_spat, F_spec, F_phys, cbar = _inputs()
    _, _, w = PrecisionGatedFusion()(F_spat, F_spec, F_phys, cbar)
    assert (w >= 0).all()
    s = w.sum(1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)


def test_07_interfaz_uniforme_variantes():
    F_spat, F_spec, F_phys, cbar = _inputs(B=1, H=12, W=12)
    for cls in VARIANTS:
        fused, c_fused, _ = cls()(F_spat, F_spec, F_phys, cbar)
        assert fused.shape == (1, 384, 12, 12)
        assert (c_fused >= 0).all() and (c_fused <= 1).all()
