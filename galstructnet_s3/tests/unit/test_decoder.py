"""Tests de specs/30_decoder_unet.md. Std (Hito 3); variante N en Hito 4."""
import pytest

torch = pytest.importorskip("torch")
from galstructnet_s3.models.decoder import FPNDecoder, FPNDecoderN


def _inputs(B=2, H=34, W=34, seed=0):
    torch.manual_seed(seed)
    features = torch.randn(B, 384, H, W)
    skips = [torch.randn(B, 256, max(1, H // 2 ** (i + 1)),
                         max(1, W // 2 ** (i + 1))) for i in range(3)]
    c_fused = torch.rand(B, 1, H, W)
    return features, skips, c_fused


@pytest.mark.parametrize("cls", [FPNDecoder, FPNDecoderN])
@pytest.mark.parametrize("hw", [(34, 34), (72, 72), (74, 74)])
def test_01_shapes_dinamicos_34_72_74(cls, hw):
    H, W = hw
    features, skips, c_fused = _inputs(H=H, W=W)
    c_skips = [torch.rand(2, 1, *s.shape[-2:]) for s in skips]
    hidden, c_dec = cls()(features, skips, c_fused, c_skips)
    assert hidden.shape == (2, 256, H, W)
    assert c_dec.shape == (2, 1, H, W)


def test_02_determinismo_eval_b1():
    features, skips, c_fused = _inputs(B=1)
    dec = FPNDecoder().eval()
    with torch.no_grad():
        a, _ = dec(features, skips, c_fused)
        b, _ = dec(features, skips, c_fused)
    assert torch.equal(a, b)


def test_03_skips_contribuyen():
    features, skips, c_fused = _inputs()
    dec = FPNDecoder().eval()
    with torch.no_grad():
        base, _ = dec(features, skips, c_fused)
        skips2 = [s.clone() for s in skips]
        skips2[1] = skips2[1] + 5.0
        pert, _ = dec(features, skips2, c_fused)
    assert not torch.allclose(base, pert)


def test_04_ignorancia_variante_n():
    # region con c_fused=0 y c_skips=0: el output alli no depende de la senal
    torch.manual_seed(0)
    dec = FPNDecoderN().eval()
    features, skips, c_fused = _inputs()
    H, W = features.shape[-2:]
    c_fused = c_fused.clone()
    c_fused[:, :, :10, :10] = 0.0
    c_skips = []
    for s in skips:
        cs = torch.rand(2, 1, *s.shape[-2:])
        h2, w2 = s.shape[-2:]
        cs[:, :, :max(1, 10 * h2 // H), :max(1, 10 * w2 // W)] = 0.0
        c_skips.append(cs)
    feats2 = features.clone()
    feats2[:, :, :10, :10] = 999.0
    skips2 = []
    for s in skips:
        s2 = s.clone()
        h2, w2 = s.shape[-2:]
        s2[:, :, :max(1, 10 * h2 // H), :max(1, 10 * w2 // W)] = 999.0
        skips2.append(s2)
    with torch.no_grad():
        ya, ca = dec(features, skips, c_fused, c_skips)
        yb, cb = dec(feats2, skips2, c_fused, c_skips)
    # el interior de la region apagada no cambia (los bordes reciben
    # informacion de vecinos validos: se excluyen del assert)
    assert torch.allclose(ya[:, :, :6, :6], yb[:, :, :6, :6], atol=1e-4)
    assert torch.allclose(ca, cb)


def test_05_c_dec_en_rango():
    features, skips, c_fused = _inputs()
    # std: c_dec = c_fused pasa-traves (interfaz uniforme)
    _, c_dec = FPNDecoder()(features, skips, c_fused)
    assert torch.equal(c_dec, c_fused)
    # N: c_dec en [0,1], 0 solo donde todo el soporte es 0
    c_skips = [torch.rand(2, 1, *s.shape[-2:]) for s in skips]
    _, c_n = FPNDecoderN()(features, skips, c_fused, c_skips)
    assert (c_n >= 0).all() and (c_n <= 1).all()
