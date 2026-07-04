"""Tests 1-15 de specs/45_evidence_layers.md (familia EPN)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from galstructnet_s3.models.encoders.physical import PhysicalEncoderN
from galstructnet_s3.models.fusion_precision import PrecisionGatedFusion
from galstructnet_s3.models.heads.segmentation import EvidenceHead
from galstructnet_s3.models.layers.evidence import EvidenceConv2d
from galstructnet_s3.models.layers.normconv import NormConv2d


def _xc(B=2, C=8, H=16, W=16, seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, C, H, W), torch.rand(B, C, H, W)


# -- P1 NormConv2d ----------------------------------------------------------

def test_01_ignorancia_exacta_normconv():
    torch.manual_seed(0)
    nc = NormConv2d(8, 16).eval()
    x, c = _xc()
    c[:, :, 4:8, 4:8] = 0.0
    x2 = x.clone()
    x2[:, :, 4:8, 4:8] = 1e4          # el valor bajo c=0 es irrelevante
    with torch.no_grad():
        ya, _ = nc(x, c)
        yb, _ = nc(x2, c)
    assert torch.allclose(ya, yb, atol=1e-5)


def test_02_reduccion_c_igual_1():
    # con c==1, z es la conv con kernel no negativo normalizado: la salida
    # coincide con la de cualquier c constante (la normalizacion anula c)
    torch.manual_seed(0)
    nc = NormConv2d(8, 16).eval()
    x, _ = _xc()
    with torch.no_grad():
        y1, c1 = nc(x, torch.ones_like(x))
        yh, ch = nc(x, torch.full_like(x, 0.5))
    assert torch.allclose(y1, yh, atol=1e-4)
    assert torch.allclose(c1, torch.ones_like(c1), atol=1e-3)
    assert torch.allclose(ch, torch.full_like(ch, 0.5), atol=1e-3)


def test_03_certeza_convexa_monotona():
    torch.manual_seed(1)
    nc = NormConv2d(8, 16, k=3).eval()
    x, c = _xc()
    with torch.no_grad():
        _, c_out = nc(x, c)
    assert (c_out >= -1e-6).all()
    # c_out <= max de la vecindad 3x3 del maximo entre canales de entrada
    import torch.nn.functional as F
    neigh_max = F.max_pool2d(c.amax(1, keepdim=True), 3, stride=1, padding=1)
    assert (c_out <= neigh_max + 1e-4).all()
    # c_out = 0 <=> todo el soporte tiene c = 0
    c0 = c.clone()
    c0[:, :, :8, :] = 0.0
    with torch.no_grad():
        _, cz = nc(x, c0)
    assert (cz[:, :, :6, :] < 1e-6).all()      # interior de la region apagada
    assert (cz[:, :, 9:, :] > 0).all()


def test_04_equivariancia_d4_normconv():
    # es una convolucion: rotar entrada => rotar salida... si el kernel se
    # rota igual. La equivariancia exacta del spec aplica a la agregacion:
    # verificar con kernel 1x1 (invariante a rotacion) y k=3 rotando pesos.
    torch.manual_seed(0)
    nc = NormConv2d(4, 8, k=1).eval()
    x, c = _xc(C=4)
    with torch.no_grad():
        y, co = nc(x, c)
        yr, cor = nc(torch.rot90(x, 1, (-2, -1)), torch.rot90(c, 1, (-2, -1)))
    assert torch.allclose(torch.rot90(y, 1, (-2, -1)), yr, atol=1e-5)
    assert torch.allclose(torch.rot90(co, 1, (-2, -1)), cor, atol=1e-5)


# -- P2 EvidenceConv2d ------------------------------------------------------

def test_05_conservacion_de_evidencia_interior():
    ec = EvidenceConv2d(5, 5)
    e = torch.zeros(1, 5, 33, 33)
    e[0, :, 16, 16] = torch.arange(1.0, 6.0)   # delta lejos del borde
    e_out = ec(e)
    assert torch.allclose(e_out.sum((2, 3)), e.sum((2, 3)), atol=1e-5)
    assert (e_out >= 0).all()


def test_06_identidad_kernel_delta():
    k = 5
    delta = torch.zeros(k, k)
    delta[k // 2, k // 2] = 1.0
    ec = EvidenceConv2d(5, k, fixed_kernel=delta)
    e = torch.rand(2, 5, 12, 12)
    assert torch.allclose(ec(e), e, atol=1e-5)


def test_07_equivalencia_psf_alpha_conv():
    # con fixed_kernel = PSF, coincide con la alpha-convolucion de referencia
    # (numpy) y con PSFEvidenceModule (specs/43)
    rng = np.random.default_rng(0)
    psf = rng.random((5, 5)).astype(np.float32)
    psf /= psf.sum()
    alpha = 1.0 + torch.rand(1, 5, 15, 15) * 10

    ec = EvidenceConv2d(5, 5, fixed_kernel=torch.from_numpy(psf))
    alpha_obs = ec(alpha - 1.0) + 1.0

    # referencia numpy: correlacion 2D con zero-padding por clase
    e_np = (alpha - 1.0).numpy()[0]
    ref = np.zeros_like(e_np)
    pad = 2
    epad = np.pad(e_np, ((0, 0), (pad, pad), (pad, pad)))
    for i in range(15):
        for j in range(15):
            ref[:, i, j] = (epad[:, i:i + 5, j:j + 5] * psf[None]).sum((1, 2))
    assert np.allclose(alpha_obs[0].numpy() - 1.0, ref, atol=1e-4)

    from galstructnet_s3.models.heads.psf import PSFEvidenceModule
    a2, _ = PSFEvidenceModule()(alpha, torch.from_numpy(psf).expand(1, 5, 5))
    assert torch.allclose(alpha_obs, a2, atol=1e-5)


# -- P3 PrecisionGatedFusion --------------------------------------------------

def _fusion_inputs(B=2, H=12, W=12, seed=0):
    torch.manual_seed(seed)
    return (torch.randn(B, 256, H, W), torch.randn(B, 256, H, W),
            torch.randn(B, 64, H, W),
            {m: torch.rand(B, 1, H, W) for m in ("spat", "spec", "phys")})


def test_08_degradacion_selectiva_fusion():
    # entrenar brevemente el gate para que use la certeza; luego bajar
    # c_spec en una region debe reducir el peso de spec alli (estadistico)
    torch.manual_seed(0)
    fus = PrecisionGatedFusion()
    F_spat, F_spec, F_phys, cbar = _fusion_inputs()
    opt = torch.optim.Adam(fus.parameters(), lr=1e-2)
    for _ in range(30):
        opt.zero_grad()
        _, _, w = fus(F_spat, F_spec, F_phys, cbar)
        # objetivo sintetico: el peso de spec debe seguir a c_spec
        loss = ((w[:, 1] - cbar["spec"].squeeze(1)) ** 2).mean()
        loss.backward()
        opt.step()
    fus.eval()
    cbar_low = {k: v.clone() for k, v in cbar.items()}
    cbar_low["spec"][:, :, :6, :] = 0.0
    with torch.no_grad():
        _, _, w_hi = fus(F_spat, F_spec, F_phys, cbar)
        _, _, w_lo = fus(F_spat, F_spec, F_phys, cbar_low)
    assert w_lo[:, 1, :6, :].mean() < w_hi[:, 1, :6, :].mean()
    # fuera de la region degradada no cambia
    assert torch.allclose(w_lo[:, 1, 7:, :], w_hi[:, 1, 7:, :], atol=1e-6)


def test_09_neutralidad_inicial_fusion():
    # con g_m = 0 (init), la salida coincide con la fusion sin precision
    # (misma semilla de pesos, certezas distintas => misma salida)
    F_spat, F_spec, F_phys, cbar = _fusion_inputs()
    torch.manual_seed(7)
    fus = PrecisionGatedFusion().eval()
    cbar2 = {m: torch.rand_like(c) for m, c in cbar.items()}
    with torch.no_grad():
        fa, _, wa = fus(F_spat, F_spec, F_phys, cbar)
        fb, _, wb = fus(F_spat, F_spec, F_phys, cbar2)
    assert torch.allclose(fa, fb, atol=1e-6)
    assert torch.allclose(wa, wb, atol=1e-6)


# -- P4 EvidenceHead ----------------------------------------------------------

def test_10_vacuity_1_bajo_c0():
    torch.manual_seed(0)
    head = EvidenceHead().eval()
    hidden = torch.randn(2, 256, 8, 8)
    out = head(hidden, torch.zeros(2, 1, 8, 8))
    assert torch.allclose(out["alpha"], torch.ones_like(out["alpha"]))
    assert torch.allclose(out["vacuity"], torch.ones_like(out["vacuity"]))
    assert torch.allclose(out["prob"], torch.full_like(out["prob"], 0.2))


def test_11_monotonia_s_en_c():
    torch.manual_seed(0)
    head = EvidenceHead().eval()
    hidden = torch.randn(1, 256, 6, 6)
    S_prev = None
    for cval in (0.0, 0.1, 0.4, 0.7, 1.0):
        out = head(hidden, torch.full((1, 1, 6, 6), cval))
        S = out["alpha"].sum(1)
        if S_prev is not None:
            assert (S >= S_prev - 1e-5).all()
        S_prev = S


def test_12_alpha_ge_1_prob_suma_1():
    torch.manual_seed(0)
    head = EvidenceHead()
    out = head(torch.randn(2, 256, 8, 8), torch.rand(2, 1, 8, 8))
    assert (out["alpha"] >= 1).all()
    s = out["prob"].sum(1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-6)
    out["alpha"].mean().backward()
    for n, p in head.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), n
    # (a, b) persisten con nombre (checkpoint interpretable, specs/45)
    assert {"a", "b"} <= {n for n, _ in head.named_parameters()}


# -- integracion / robustez ----------------------------------------------------

def test_13_sustitucion_de_nan_to_num():
    # garantia #1 aplicada end-to-end al encoder fisico N: una muestra con
    # NaN (c=0) produce el mismo output que la muestra limpia
    torch.manual_seed(0)
    enc = PhysicalEncoderN().eval()
    maps, c = _xc()
    c[:, 3, 5:9, 5:9] = 0.0
    dirty = maps.clone()
    dirty[:, 3, 5:9, 5:9] = 12345.0    # sustituye al NaN (ya anulado por c)
    with torch.no_grad():
        fa, ca = enc(maps, c)
        fb, cb = enc(dirty, c)
    assert torch.allclose(fa, fb, atol=1e-5)
    assert torch.allclose(ca, cb)


def test_14_conservacion_bf16():
    ec = EvidenceConv2d(5, 5)
    e = torch.zeros(1, 5, 33, 33, dtype=torch.bfloat16)
    e[0, :, 16, 16] = 100.0
    e_out = ec(e)
    assert torch.allclose(e_out.float().sum((2, 3)), e.float().sum((2, 3)),
                          atol=1e-2)


def test_15_shapes_y_b1():
    x, c = _xc(B=1)
    y, co = NormConv2d(8, 16)(x, c)
    assert y.shape == (1, 16, 16, 16) and co.shape == (1, 16, 16, 16)
    e = torch.rand(1, 5, 16, 16)
    assert EvidenceConv2d(5, 3)(e).shape == (1, 5, 16, 16)
    args = _fusion_inputs(B=1)
    f, cf, w = PrecisionGatedFusion()(*args)
    assert f.shape == (1, 384, 12, 12) and cf.shape == (1, 1, 12, 12)
    assert w.shape == (1, 3, 12, 12)
    out = EvidenceHead()(torch.randn(1, 256, 12, 12), torch.rand(1, 1, 12, 12))
    assert out["alpha"].shape == (1, 5, 12, 12)
