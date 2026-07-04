"""Tests 1-8 de specs/10_dataset.md sobre la fixture synthetic_entry.

El contrato es size-agnostic: ningun test asume H=W=34 mas alla de leerlo
del propio archivo.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from galstructnet_s3.data import GalStructDataset, collate_pad, pad_to_multiple
from galstructnet_s3.data.transforms import (apply_dihedral, ChannelDropout,
                                             invert_dihedral, restframe_shift)

CONTRACT_SPATIAL = {
    # clave: canales esperados (None = tensor 2D (H, W))
    "cube": "L", "c_spec": 1, "image": 3, "c_spat": 3, "maps": 8, "c_phys": 8,
    "Y_mass": 5, "Y_mass_obs": 5, "Y_lum": 5, "Y_lum_obs": 5,
    "n_eff_mass": None, "n_eff_mass_obs": None,
    "n_eff_lum": None, "n_eff_lum_obs": None,
    "M": None, "M_unc_mass": None, "M_unc_lum": None, "w_phys_mass": None,
}


@pytest.fixture()
def ds(synthetic_entry):
    return GalStructDataset(synthetic_entry.parent, "train")


@pytest.fixture()
def sample(ds):
    return ds[0]


def test_01_shapes_del_contrato_sin_literales(sample, synthetic_entry):
    with h5py.File(synthetic_entry, "r") as f:
        H, W = f["masks/M_valid"].shape
        L = f["inputs/cube_ifu"].shape[0]
    for key, ch in CONTRACT_SPATIAL.items():
        t = sample[key]
        assert t.shape[-2:] == (H, W), key
        if ch is None:
            assert t.dim() == 2, key
        else:
            assert t.shape[0] == (L if ch == "L" else ch), key
    K = sample["psf"].shape[-1]
    assert sample["psf"].shape == (K, K) and K % 2 == 1
    assert abs(sample["psf"].sum().item() - 1.0) < 1e-5
    for w in ("mass", "lum"):
        assert sample[f"target_fractions_{w}"].shape == (5,)
    assert isinstance(sample["galaxy_id"], str)
    assert isinstance(sample["view_id"], int)
    assert sample["hw_native"] == (H, W)


def test_02_etiquetas_suman_1_atol_1e_2(sample):
    M = sample["M"]
    for key in ("Y_mass", "Y_mass_obs", "Y_lum", "Y_lum_obs"):
        s = sample[key].sum(0)[M]
        assert torch.allclose(s, torch.ones_like(s), atol=1e-2), key


def test_03_certezas_en_rango_y_c0_donde_invalido(synthetic_entry):
    # inyecta NaN en maps para verificar c=0 exactamente ahi (y en ~M_valid)
    with h5py.File(synthetic_entry, "r+") as f:
        maps = f["inputs/pipe3d_maps"][()]
        H, W = maps.shape[-2:]
        maps[2, H // 2, W // 2] = np.nan
        f["inputs/pipe3d_maps"][...] = maps
    s = GalStructDataset(synthetic_entry.parent, "train")[0]
    M = s["M"]
    for key in ("c_spec", "c_spat", "c_phys"):
        c = s[key]
        assert (c >= 0).all() and (c <= 1).all(), key
        assert (c[..., ~M] == 0).all(), key
    assert s["c_phys"][2, H // 2, W // 2] == 0.0
    assert torch.isfinite(s["maps"]).all()
    # donde es valido y sin NaN, la certeza es positiva
    assert (s["c_phys"][3][M] > 0).all()


def test_04_anclas_coherentes_n_eff(sample):
    for key in ("n_eff_mass", "n_eff_lum", "n_eff_mass_obs", "n_eff_lum_obs"):
        n = sample[key]
        assert (n >= 0).all() and torch.isfinite(n).all(), key
    # spaxels con N_eff_raw == 0 no pueden ser validos (MaskBuilder)
    assert not sample["M"][sample["n_eff_mass"] == 0].any()


def test_05_padding_y_collate_mixto(ds):
    s1 = pad_to_multiple(ds[0], mult=32)
    H1, W1 = s1["M"].shape
    assert H1 % 32 == 0 and W1 % 32 == 0
    h0, w0 = s1["hw_native"]
    assert (~s1["M"][h0:, :]).all() and (s1["c_phys"][..., :, w0:] == 0).all()

    # batch mixto: el segundo sample simula un bundle menor (crop nativo)
    s2 = ds[0]
    hw_full = tuple(s2["M"].shape)
    for k, v in list(s2.items()):
        if (torch.is_tensor(v) and v.dim() >= 2
                and tuple(v.shape[-2:]) == hw_full and k != "psf"):
            s2[k] = v[..., :-8, :-4]
    s2["hw_native"] = tuple(s2["M"].shape)
    s2 = pad_to_multiple(s2, mult=16)
    s1b = pad_to_multiple(ds[0], mult=16)

    batch = collate_pad([s1b, s2])
    B, H, W = batch["M"].shape
    assert B == 2
    assert H == max(s1b["M"].shape[0], s2["M"].shape[0])
    assert W == max(s1b["M"].shape[1], s2["M"].shape[1])
    assert batch["cube"].shape[-2:] == (H, W)
    assert batch["hw_native"] == [s1b["hw_native"], s2["hw_native"]]
    # padding de batch tambien ignorable: M=False, c=0
    assert not batch["M"][1, s2["M"].shape[0]:, :].any()
    assert (batch["c_spat"][1, ..., s2["M"].shape[1]:] == 0).all()


def test_06_dihedral_mantiene_alineacion(ds):
    s = ds[0]
    ref = {k: v.clone() for k, v in s.items() if torch.is_tensor(v)}
    for k in range(4):
        for flip in (False, True):
            out = invert_dihedral(apply_dihedral(dict(s), k, flip), k, flip)
            for key, v in ref.items():
                assert torch.equal(out[key], v), (key, k, flip)
    # alineacion senal-certeza-etiqueta: rotar mantiene la correspondencia
    rot = apply_dihedral({k: v.clone() for k, v in s.items()
                          if torch.is_tensor(v)}, 1, False)
    assert torch.equal(rot["M"], torch.rot90(s["M"], 1, dims=(-2, -1)))
    assert torch.equal(rot["Y_lum"], torch.rot90(s["Y_lum"], 1, dims=(-2, -1)))
    assert torch.equal(rot["c_phys"], torch.rot90(s["c_phys"], 1, dims=(-2, -1)))
    # determinismo del pipeline completo con seed
    ds_a = GalStructDataset(ds.root, "train", augment=True, seed=7)
    ds_b = GalStructDataset(ds.root, "train", augment=True, seed=7)
    a, b = ds_a[0], ds_b[0]
    for key in ("cube", "c_phys", "Y_mass"):
        assert torch.equal(a[key], b[key]), key


def test_07_channel_dropout_apaga_c_phys_h3h4(sample):
    rng = np.random.default_rng(0)
    maps_before = sample["maps"].clone()
    out = ChannelDropout(p=1.0)(sample, rng)
    assert (out["c_phys"][6:8] == 0).all()
    assert torch.equal(out["maps"], maps_before)  # senal intacta
    assert (out["c_phys"][:6][..., out["M"]] > 0).any()


def test_08_restframe_identidad_con_v0(sample):
    L = sample["cube"].shape[0]
    H, W = sample["M"].shape
    cube = torch.randn(L, H, W)
    v0 = torch.zeros(H, W)
    c1 = torch.ones(H, W)
    assert torch.allclose(restframe_shift(cube, v0, c1, 1e-4), cube)

    # linea sintetica observada en el canal m con v>0 (redshift): en
    # rest-frame cae n_chan canales hacia el azul (rejilla log10-lineal)
    dloglam = 1e-4
    n_chan = 5
    v = (10 ** (n_chan * dloglam) - 1.0) * 299_792.458
    m = L // 2
    line = torch.zeros(L, H, W)
    line[m] = 1.0
    out = restframe_shift(line, torch.full((H, W), v), c1, dloglam)
    assert out[m - n_chan].mean() > 0.9
    assert out[m].mean() < 0.1
    # con certeza baja en v no se corrige
    out_nc = restframe_shift(line, torch.full((H, W), v), torch.zeros(H, W),
                             dloglam)
    assert torch.allclose(out_nc, line)
