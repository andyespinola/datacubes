"""Fixtures compartidas. La pieza central es `synthetic_entry`: un
`dataset_entry` HDF5 sintético que codifica EJECUTABLEMENTE el contrato de
datos v3 (specs/10_dataset.md). Dimensiones reducidas (H=W=34, L=64) — el
código bajo test no debe hardcodear tamaños, así que sirve igual.

El piloto real es TNG50-87-141934-0-127; esta fixture lo sustituye en CI.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def synthetic_entry(tmp_path):
    """Crea dataset_entry_SYNTH-0000_v0.h5 con el contrato v3 completo."""
    pytest.importorskip("h5py")
    return make_synthetic_entry(tmp_path)


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    """Directorio de sesion con un entry sintetico (tests de integracion,
    que no mutan el archivo)."""
    pytest.importorskip("h5py")
    root = tmp_path_factory.mktemp("entries")
    make_synthetic_entry(root)
    return root


def make_synthetic_entry(dirpath):
    import h5py
    rng = np.random.default_rng(42)
    H = W = 34
    L = 64          # reducido (real: 6603); nada debe depender del valor
    K = 5

    path = dirpath / "dataset_entry_SYNTH-0000_v0.h5"

    # geometría: disco válido central + perfiles radiales
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(yy - H / 2, xx - W / 2)
    M_valid = r < (H / 2 - 2)

    def simplex(shape_hw, conc):
        a = rng.gamma(conc, size=(K, *shape_hw)).astype(np.float32)
        return a / a.sum(axis=0, keepdims=True)

    with h5py.File(path, "w") as f:
        g = f.create_group("inputs")
        g.create_dataset("cube_ifu", data=rng.normal(
            1.0, 0.1, size=(L, H, W)).astype(np.float32))
        snr = np.clip(30.0 * np.exp(-r / 10.0), 0.5, None).astype(np.float32)
        g.create_dataset("snr_spec", data=snr)
        g.create_dataset("image", data=rng.normal(
            0, 1, size=(3, H, W)).astype(np.float32))
        maps = rng.normal(0, 1, size=(8, H, W)).astype(np.float32)
        maps[0] += (xx - W / 2) * 0.2          # v_star: gradiente (rotación)
        g.create_dataset("pipe3d_maps", data=maps)
        err = np.abs(rng.normal(0.3, 0.05, size=(8, H, W))).astype(np.float32)
        err[6:8] *= 3.0                         # h3/h4 (pPXF) más ruidosos
        g.create_dataset("pipe3d_err", data=err)
        ker = np.exp(-(np.hypot(*np.mgrid[-3:4, -3:4])) ** 2 / 2.0)
        g.create_dataset("psf_kernel", data=(ker / ker.sum()).astype(np.float32))

        lab = f.create_group("labels")
        neff = np.where(M_valid, 5.0 + 400.0 * np.exp(-r / 6.0), 0.0)
        for weight in ("mass", "lum"):
            Y_raw = simplex((H, W), conc=2.0)
            Y_raw[:, ~M_valid] = 1.0 / K
            Y_psf = Y_raw * 0.9 + (1.0 / K) * 0.1      # proxy suavizado
            lab.create_dataset(f"Y_{weight}_raw", data=Y_raw.astype(np.float32))
            lab.create_dataset(f"Y_{weight}_psf",
                               data=(Y_psf / Y_psf.sum(0)).astype(np.float32))
            lab.create_dataset(f"N_eff_raw_{weight}", data=neff.astype(np.float32))
            lab.create_dataset(f"N_eff_psf_{weight}",
                               data=(neff * 0.9).astype(np.float32))
        lab.create_dataset("class_names", data=np.array(
            ["bulge", "disk", "bar", "arm", "other"], dtype="S"))

        m = f.create_group("masks")
        m.create_dataset("M_valid", data=M_valid)
        m.create_dataset("M_uncertain_mass", data=(neff > 0) & (neff < 15))
        m.create_dataset("M_uncertain_lum", data=(neff > 0) & (neff < 15))

        qa = f.create_group("qa")
        for weight in ("mass", "lum"):
            fr = rng.dirichlet(np.ones(K)).astype(np.float32)
            qa.attrs[f"target_fractions_{weight}"] = fr

    return path
