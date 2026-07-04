"""Consistencia estadistica D4 del modelo completo (specs/21 v3, C8).

Swin no es equivariante: el test valido es rotar el INPUT completo
(senal+certeza+mascaras), forwardear, des-rotar la prediccion y comparar
con el forward original CON TOLERANCIA estadistica. Con el modelo
entrenado la tolerancia se endurece; aqui (pesos aleatorios) se verifica
que el pipeline de rotacion/des-rotacion esta bien cableado y que la
discrepancia es acotada.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("h5py")
pytest.importorskip("timm")

from torch.utils.data import DataLoader

from galstructnet_s3.data import GalStructDataset, collate_pad
from galstructnet_s3.data.transforms import SPATIAL_KEYS
from galstructnet_s3.models.model import build_model

CFG = {"model": {"spatial": "std", "physical": "std", "fusion": "concat",
                 "decoder": "std", "head": "std",
                 "spectral": {"backbone": "conv1d", "d_model": 32,
                              "d_out": 256, "n_layers": 2}}}


def _rot_batch(batch: dict, k: int) -> dict:
    out = dict(batch)
    for key in SPATIAL_KEYS:
        if key in out and torch.is_tensor(out[key]):
            out[key] = torch.rot90(out[key], k, dims=(-2, -1))
    return out


def test_01_consistencia_d4_del_modelo_completo_con_tolerancia(synthetic_root):
    torch.manual_seed(0)
    model = build_model(CFG).eval()
    ds = GalStructDataset(synthetic_root, "train", mult=32)
    batch = next(iter(DataLoader(ds, batch_size=1, collate_fn=collate_pad)))

    with torch.no_grad():
        p0 = model(batch)["lum"]["prob"]
        p90 = model(_rot_batch(batch, 1))["lum"]["prob"]
    p90_back = torch.rot90(p90, -1, dims=(-2, -1))

    M = batch["M"]
    diff = (p0 - p90_back).abs().sum(1)[M]      # L1 en el simplex por spaxel
    # tolerancia estadistica: mediana de discrepancia acotada (el argmax y
    # las metricas downstream son estables); se endurece post-entrenamiento
    assert float(diff.median()) < 0.15, float(diff.median())
    assert torch.isfinite(p90_back).all()
