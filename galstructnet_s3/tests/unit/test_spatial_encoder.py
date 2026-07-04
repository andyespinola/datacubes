"""Tests de specs/21_encoder_spatial.md (parche v3): shapes, mult expuesto,
determinismo, sensibilidad espacial. Sin test de equivariancia (C8)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
from galstructnet_s3.models.encoders.spatial import SwinSpatialEncoder


@pytest.fixture(scope="module")
def enc():
    return SwinSpatialEncoder(pretrained=False).eval()


def test_01_shapes_y_skips(enc):
    for H, W in ((64, 64), (96, 64)):
        img = torch.randn(2, 3, H, W)
        with torch.no_grad():
            feats, skips = enc(img)
        assert feats.shape == (2, 256, H, W)
        assert len(skips) == 3
        for i, s in enumerate(skips):
            assert s.shape[:2] == (2, 256)
            assert s.shape[-2:] == (H // 2 ** (i + 1), W // 2 ** (i + 1))


def test_02_mult_expuesto(enc):
    assert enc.mult == 32   # patch 4 * 2**3; el dataset lo lee de config


def test_03_determinismo(enc):
    img = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        a, _ = enc(img)
        b, _ = enc(img)
    assert torch.equal(a, b)


def test_04_sensibilidad_espacial(enc):
    # cambiar un pixel central afecta una vecindad amplia (>5x5): el encoder
    # mezcla informacion espacial (su razon de existir, specs/21)
    img = torch.randn(1, 3, 64, 64)
    img2 = img.clone()
    img2[0, :, 32, 32] += 10.0
    with torch.no_grad():
        a, _ = enc(img)
        b, _ = enc(img2)
    diff = (a - b).abs().sum(1)[0]
    assert (diff > 1e-6).sum() > 25


def test_05_gradientes_finitos():
    enc = SwinSpatialEncoder(pretrained=False)
    feats, skips = enc(torch.randn(1, 3, 64, 64))
    (feats.mean() + sum(s.mean() for s in skips)).backward()
    for n, p in enc.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), n
