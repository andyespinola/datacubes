"""Tests de specs/20_encoder_spectral.md (parche v3).

En CPU/CI corre DilatedConv1DEncoder (contrafactual A8, misma interfaz);
los tests de Mamba se activan solo con mamba_ssm + CUDA disponibles.
"""
import pytest

torch = pytest.importorskip("torch")
from galstructnet_s3.models.encoders.spectral import (DilatedConv1DEncoder,
                                                      build_spectral_encoder)

HAS_MAMBA = False
try:
    import mamba_ssm  # noqa: F401
    HAS_MAMBA = torch.cuda.is_available()
except ImportError:
    pass


def _cube(B=2, L=64, H=6, W=6, seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, L, H, W)


def _encoders():
    encs = [DilatedConv1DEncoder(d_model=32, d_out=48, n_layers=2)]
    if HAS_MAMBA:
        encs.append(build_spectral_encoder(
            "mamba", d_model=32, d_out=48, n_layers=2).cuda())
    return encs


def test_01_shape_salida():
    cube = _cube()
    for enc in _encoders():
        x = cube.cuda() if next(enc.parameters()).is_cuda else cube
        out = enc(x)
        assert out.shape == (2, 48, 6, 6)


def test_03_sin_gradiente_espacial_entre_spaxels():
    # cada spaxel es independiente: perturbar (i',j') no cambia (i,j)
    enc = DilatedConv1DEncoder(d_model=32, d_out=48, n_layers=2).eval()
    cube = _cube(B=1)
    with torch.no_grad():
        base = enc(cube)
        cube2 = cube.clone()
        cube2[0, :, 0, 0] += 10.0
        pert = enc(cube2)
    assert not torch.allclose(base[0, :, 0, 0], pert[0, :, 0, 0])
    assert torch.allclose(base[0, :, 3, 3], pert[0, :, 3, 3], atol=1e-6)


def test_04_determinismo_eval_y_b1():
    enc = DilatedConv1DEncoder(d_model=32, d_out=48, n_layers=2).eval()
    cube = _cube(B=1)
    with torch.no_grad():
        assert torch.equal(enc(cube), enc(cube))


def test_06_return_sequence_sin_pooling():
    # Etapa 1 (C6): secuencia (N, L', d_model) para SpectralMAEHead
    enc = DilatedConv1DEncoder(d_model=32, d_out=48, n_layers=2,
                               return_sequence=True).eval()
    B, L, H, W = 1, 64, 3, 3
    seq = enc(_cube(B=B, L=L, H=H, W=W))
    assert seq.shape == (B * H * W, L // 4, 32)


def test_07_gradientes_finitos():
    enc = DilatedConv1DEncoder(d_model=32, d_out=48, n_layers=2)
    out = enc(_cube(B=1, H=4, W=4))
    out.mean().backward()
    for n, p in enc.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), n


@pytest.mark.skipif(not HAS_MAMBA, reason="mamba_ssm + CUDA no disponibles")
def test_08_mamba_equivale_interfaz():
    enc = build_spectral_encoder("mamba", d_model=32, d_out=48,
                                 n_layers=2).cuda().eval()
    with torch.no_grad():
        out = enc(_cube(B=1).cuda())
    assert out.shape == (1, 48, 6, 6) and torch.isfinite(out).all()
    seq_enc = build_spectral_encoder("mamba", d_model=32, d_out=48,
                                     n_layers=2, return_sequence=True).cuda()
    seq = seq_enc(_cube(B=1, H=3, W=3).cuda())
    assert seq.shape == (9, 16, 32)
