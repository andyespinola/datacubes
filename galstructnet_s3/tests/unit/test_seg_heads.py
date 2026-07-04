"""Tests 1-7 de specs/40_head_segmentation.md (cabezas duales, std y EPN)."""
import pytest

torch = pytest.importorskip("torch")
from galstructnet_s3.models.heads.segmentation import DualSegHeads

KINDS = ("std", "evidence")


def _inputs(B=2, H=10, W=14, seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, 256, H, W), torch.rand(B, 1, H, W)


@pytest.mark.parametrize("kind", KINDS)
def test_01_shapes_8_tensores(kind):
    hidden, c = _inputs()
    out = DualSegHeads(kind=kind)(hidden, c)
    for t in ("mass", "lum"):
        for key, ch in (("alpha", 5), ("prob", 5), ("evidence", 5),
                        ("vacuity", 1)):
            assert out[t][key].shape == (2, ch, 10, 14), (t, key)


@pytest.mark.parametrize("kind", KINDS)
def test_02_invariantes_alpha_prob_vacuity(kind):
    hidden, c = _inputs()
    out = DualSegHeads(kind=kind)(hidden, c)
    for t in ("mass", "lum"):
        assert (out[t]["alpha"] >= 1).all()
        s = out[t]["prob"].sum(1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-6)
        v = out[t]["vacuity"]
        assert (v > 0).all() and (v <= 1 + 1e-6).all()


def test_03_vacuity_bajo_ignorancia():
    hidden, _ = _inputs()
    out = DualSegHeads(kind="evidence")(hidden, torch.zeros(2, 1, 10, 14))
    for t in ("mass", "lum"):
        assert torch.allclose(out[t]["alpha"], torch.ones_like(out[t]["alpha"]))
        assert torch.allclose(out[t]["vacuity"],
                              torch.ones_like(out[t]["vacuity"]))


def test_04_monotonia_en_certeza():
    hidden, _ = _inputs(B=1)
    heads = DualSegHeads(kind="evidence").eval()
    S_prev = None
    for cv in (0.0, 0.25, 0.5, 0.75, 1.0):
        out = heads(hidden, torch.full((1, 1, 10, 14), cv))
        S = out["lum"]["alpha"].sum(1)
        if S_prev is not None:
            assert (S >= S_prev - 1e-5).all()
        S_prev = S


@pytest.mark.parametrize("kind", KINDS)
def test_05_caso_degenerado_uniforme(kind):
    hidden, c = _inputs()
    heads = DualSegHeads(kind=kind)
    with torch.no_grad():
        for head in (heads.head_mass, heads.head_lum):
            head.proj.weight.zero_()
            head.proj.bias.zero_()
    out = heads(hidden, c)
    for t in ("mass", "lum"):
        p = out[t]["prob"]
        assert torch.allclose(p, torch.full_like(p, 0.2), atol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_06_independencia_de_cabezas(kind):
    # C3: la perdida de masa no actualiza head_lum.proj (si el tronco)
    hidden, c = _inputs()
    hidden.requires_grad_(True)
    heads = DualSegHeads(kind=kind)
    out = heads(hidden, c)
    out["mass"]["alpha"].sum().backward()
    assert heads.head_lum.proj.weight.grad is None \
        or heads.head_lum.proj.weight.grad.abs().sum() == 0
    assert heads.head_mass.proj.weight.grad is not None
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", KINDS)
def test_07_determinismo_gradientes_finitos(kind):
    hidden, c = _inputs()
    heads = DualSegHeads(kind=kind)
    heads.eval()
    with torch.no_grad():
        a = heads(hidden, c)["lum"]["alpha"]
        b = heads(hidden, c)["lum"]["alpha"]
    assert torch.equal(a, b)
    out = heads(hidden, c)
    (out["lum"]["alpha"].mean() + out["mass"]["alpha"].mean()).backward()
    for n, p in heads.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), n
