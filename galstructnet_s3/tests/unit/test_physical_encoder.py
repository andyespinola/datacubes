"""Tests de specs/22_encoder_physical.md. Std (Hito 3); N/MP se activan en
Hito 4 (skips individuales, no de modulo)."""
import pytest

torch = pytest.importorskip("torch")
from galstructnet_s3.models.encoders.physical import (PhysicalEncoderN,
                                                       PhysicalEncoderStd)


def _inputs(B=2, H=34, W=34, seed=0):
    torch.manual_seed(seed)
    maps = torch.randn(B, 8, H, W)
    c = torch.rand(B, 8, H, W)
    return maps, c


@pytest.mark.parametrize("cls", [PhysicalEncoderStd, PhysicalEncoderN])
@pytest.mark.parametrize("hw", [(34, 34), (72, 72)])
def test_01_shapes_tres_variantes_hw_arbitrarios(cls, hw):
    # MP (momentos) queda como experimento lateral (specs/22)
    H, W = hw
    maps, c = _inputs(H=H, W=W)
    f, c_out = cls()(maps, c)
    assert f.shape == (2, 64, H, W)
    assert c_out.shape == (2, 1, H, W)


def test_02_ignorancia_exacta_variante_n():
    # test 13 de specs/45 aplicado aqui: randomizar maps donde c=0 no
    # cambia el output
    torch.manual_seed(0)
    enc = PhysicalEncoderN().eval()
    maps, c = _inputs()
    c[:, :, 12:20, 12:20] = 0.0
    maps2 = maps.clone()
    maps2[:, :, 12:20, 12:20] = torch.randn(2, 8, 8, 8) * 100
    with torch.no_grad():
        fa, ca = enc(maps, c)
        fb, cb = enc(maps2, c)
    assert torch.allclose(fa, fb, atol=1e-5)
    assert torch.allclose(ca, cb)


def test_03_std_sin_nan():
    maps, c = _inputs()
    c[:, :, 10:20, 10:20] = 0.0
    enc = PhysicalEncoderStd().eval()
    f, c_out = enc(maps, c)
    assert torch.isfinite(f).all() and torch.isfinite(c_out).all()
    # senal anulada donde c=0 (equivalente funcional de nan_to_num)
    maps2 = maps.clone()
    maps2[:, :, 10:20, 10:20] = 1e6           # valores absurdos bajo c=0
    f2, _ = enc(maps2, c)
    assert torch.allclose(f, f2)


def test_04_dropout_h3h4_output_finito_y_distinto():
    maps, c = _inputs()
    enc = PhysicalEncoderStd().eval()
    f_full, _ = enc(maps, c)
    c_drop = c.clone()
    c_drop[:, 6:8] = 0.0
    f_drop, _ = enc(maps, c_drop)
    assert torch.isfinite(f_drop).all()
    assert not torch.allclose(f_full, f_drop)  # h3/h4 se usan cuando existen


def test_05_determinismo_eval_b1():
    maps, c = _inputs(B=1)
    enc = PhysicalEncoderStd().eval()
    with torch.no_grad():
        a = enc(maps, c)[0]
        b = enc(maps, c)[0]
    assert torch.equal(a, b)


@pytest.mark.skip(reason="experimento lateral - variante MP (specs/22 Nivel B)")
def test_06_mp_monotonia_de_varianza():
    ...
