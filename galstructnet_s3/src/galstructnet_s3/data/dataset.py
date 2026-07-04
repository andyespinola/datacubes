"""Dataset v3. Spec: specs/10_dataset.md - Hito 1.

Contrato de salida: pares (senal, certeza) por modalidad, 4 variantes de
etiqueta + N_eff (raw/psf, mass/lum), mascaras, w_phys_mass, fracciones.
Sin literales espaciales hardcodeados: size-agnostic.

Splits: `root/splits/{split}.txt` con un galaxy_id por linea (las 4 vistas
de cada galaxia van juntas al mismo split). Si el directorio de splits no
existe (fixture sintetica/CI), el split contiene todos los entries de root.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .collate import pad_to_multiple
from .stats import load_or_compute_stats, symlog
from .transforms import (ChannelDropout, Compose, RandomDihedral,
                         SpectralJitter, restframe_shift)

_ENTRY_RE = re.compile(r"dataset_entry_(?P<gid>.+)_v(?P<view>\d+)\.h5$")

# indices en pipe3d_maps: (v, sigma, age, Z, mass, av, h3, h4)
_MASS_CH = 4
_VSTAR_CH = 0


def to_certainty(sigma: torch.Tensor, sigma_ref: torch.Tensor) -> torch.Tensor:
    """c = sigma_ref^2 / (sigma^2 + sigma_ref^2) en (0,1]; c(sigma_ref)=0.5.

    El llamador pone c=0 donde ~M_valid o NaN (specs/10, 'Certeza desde
    precision instrumental').
    """
    return sigma_ref ** 2 / (sigma ** 2 + sigma_ref ** 2)


def _zscore(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std.clamp_min(1e-6)


def check_gz3d_partition(root: str | Path) -> tuple[set[str], set[str]]:
    """Particion GZ3D weak/val DISJUNTA por galaxia (specs/60, specs/70).

    Se llama al construir loaders de Etapa 3 y en evaluacion externa;
    usar GZ3D integro para ambas cosas invalidaria la validacion.
    """
    root = Path(root)
    weak_f = root / "splits" / "manga_gz3d_weak.txt"
    val_f = root / "splits" / "manga_gz3d_val.txt"
    weak = {ln.strip() for ln in weak_f.read_text().splitlines() if ln.strip()}
    val = {ln.strip() for ln in val_f.read_text().splitlines() if ln.strip()}
    overlap = weak & val
    assert not overlap, (f"particion GZ3D invalida: {len(overlap)} galaxias "
                         f"en weak y val (p.ej. {sorted(overlap)[:3]})")
    return weak, val


class GalStructDataset(Dataset):
    """Lee dataset_entry_{gid}_v{view}.h5 y entrega el contrato v3.

    - HDF5 abierto por item (los handles no sobreviven el fork de workers).
    - Normalizacion: symlog+zscore por lambda (cube; symlog == log1p para
      flujo >= 0 y definido para flujo negativo), zscore por banda/mapa;
      pipe3d_err NO se z-scorea (solo via to_certainty); w_phys_mass = masa
      RAW clamp>=0 (C7); PSF y etiquetas sin normalizar.
    - Augmentations (solo con augment=True; deterministicas con seed por
      (epoch, idx)): RandomDihedral, SpectralJitter ~ 1/snr, ChannelDropout.
    - `mult`: multiplo espacial que declara el encoder espacial (specs/21);
      llega por config y se aplica con pad_to_multiple.
    """

    def __init__(self, root: str | Path, split: str, target: str = "both",
                 restframe: bool = False, carry_full_ivar: bool = False,
                 views_per_epoch: int = 4, mult: int = 1, augment: bool = False,
                 seed: int = 42, dloglam: float = 1e-4,
                 stats: dict | None = None) -> None:
        self.root = Path(root)
        self.split = split
        self.target = target
        self.restframe = restframe
        self.carry_full_ivar = carry_full_ivar
        self.views_per_epoch = views_per_epoch
        self.mult = mult
        self.augment = augment
        self.seed = seed
        self.dloglam = dloglam
        self._epoch = 0

        entries = sorted(self.root.glob("dataset_entry_*.h5"))
        if not entries:
            raise FileNotFoundError(f"sin dataset_entry_*.h5 en {self.root}")

        split_file = self.root / "splits" / f"{split}.txt"
        gids = None  # sin splits (fixture/CI): todos los entries
        if split_file.exists():
            gids = {ln.strip() for ln in split_file.read_text().splitlines()
                    if ln.strip()}

        self._by_galaxy: dict[str, list[tuple[int, Path]]] = {}
        for p in entries:
            m = _ENTRY_RE.search(p.name)
            if m is None:
                continue
            gid, view = m.group("gid"), int(m.group("view"))
            if gids is not None and gid not in gids:
                continue
            self._by_galaxy.setdefault(gid, []).append((view, p))
        if not self._by_galaxy:
            raise ValueError(f"split '{split}' vacio en {self.root}")
        for views in self._by_galaxy.values():
            views.sort()

        self._rebuild_index()

        self.stats = stats if stats is not None else load_or_compute_stats(
            self.root, [p for vs in self._by_galaxy.values() for _, p in vs])
        s = self.stats
        self._cube_mean = torch.tensor(s["cube_mean"], dtype=torch.float32)
        self._cube_std = torch.tensor(s["cube_std"], dtype=torch.float32)
        self._img_mean = torch.tensor(s["image_mean"],
                                      dtype=torch.float32).view(-1, 1, 1)
        self._img_std = torch.tensor(s["image_std"],
                                     dtype=torch.float32).view(-1, 1, 1)
        self._map_mean = torch.tensor(s["maps_mean"],
                                      dtype=torch.float32).view(-1, 1, 1)
        self._map_std = torch.tensor(s["maps_std"],
                                     dtype=torch.float32).view(-1, 1, 1)
        self._sigma_ref_phys = torch.tensor(s["sigma_ref_phys"],
                                            dtype=torch.float32).view(-1, 1, 1)
        self._sigma_ref_spat = (torch.tensor(s["sigma_ref_spat"],
                                             dtype=torch.float32).view(-1, 1, 1)
                                if s.get("sigma_ref_spat") else None)
        self._snr_ref = float(s["snr_ref"])

        self._augment_pipe = Compose([RandomDihedral(), SpectralJitter(),
                                      ChannelDropout()]) if augment else None

    # -- muestreo de vistas por epoca (specs/10 'Notas': views_per_epoch) ----

    def set_epoch(self, epoch: int) -> None:
        """Re-sortea que vistas de cada galaxia se exponen esta epoca."""
        self._epoch = epoch
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        index: list[tuple[str, int, Path]] = []
        for gid, views in sorted(self._by_galaxy.items()):
            chosen = views
            if self.views_per_epoch < len(views):
                rng = np.random.default_rng(np.array(
                    [self.seed, self._epoch, hash(gid) % (2 ** 31)],
                    dtype=np.uint64))
                sel = rng.choice(len(views), self.views_per_epoch, replace=False)
                chosen = [views[i] for i in sorted(sel)]
            index.extend((gid, v, p) for v, p in chosen)
        self._index = index

    def __len__(self) -> int:
        return len(self._index)

    # -- carga ---------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        import h5py

        gid, view, path = self._index[idx]
        with h5py.File(path, "r") as f:
            cube = torch.from_numpy(
                f["inputs/cube_ifu"][()].astype(np.float32))
            snr = torch.from_numpy(f["inputs/snr_spec"][()].astype(np.float32))
            image = torch.from_numpy(f["inputs/image"][()].astype(np.float32))
            image_err = (torch.from_numpy(
                f["inputs/image_err"][()].astype(np.float32))
                if "image_err" in f["inputs"] else None)
            maps = torch.from_numpy(
                f["inputs/pipe3d_maps"][()].astype(np.float32))
            perr = torch.from_numpy(
                f["inputs/pipe3d_err"][()].astype(np.float32))
            psf = torch.from_numpy(
                f["inputs/psf_kernel"][()].astype(np.float32))

            labels: dict[str, torch.Tensor] = {}
            for w in ("mass", "lum"):
                labels[f"Y_{w}"] = torch.from_numpy(
                    f[f"labels/Y_{w}_raw"][()].astype(np.float32))
                labels[f"Y_{w}_obs"] = torch.from_numpy(
                    f[f"labels/Y_{w}_psf"][()].astype(np.float32))
                labels[f"n_eff_{w}"] = torch.from_numpy(
                    f[f"labels/N_eff_raw_{w}"][()].astype(np.float32))
                labels[f"n_eff_{w}_obs"] = torch.from_numpy(
                    f[f"labels/N_eff_psf_{w}"][()].astype(np.float32))

            M = torch.from_numpy(f["masks/M_valid"][()].astype(bool))
            M_unc_mass = torch.from_numpy(
                f["masks/M_uncertain_mass"][()].astype(bool))
            M_unc_lum = torch.from_numpy(
                f["masks/M_uncertain_lum"][()].astype(bool))

            tf_mass = torch.tensor(f["qa"].attrs["target_fractions_mass"],
                                   dtype=torch.float32)
            tf_lum = torch.tensor(f["qa"].attrs["target_fractions_lum"],
                                  dtype=torch.float32)

        # certezas ANTES de normalizar (los errores se consumen crudos)
        c_phys = to_certainty(perr, self._sigma_ref_phys)
        if image_err is not None and self._sigma_ref_spat is not None:
            c_spat = to_certainty(image_err, self._sigma_ref_spat)
        else:
            c_spat = torch.ones_like(image)
        c_spec = (snr / (snr + self._snr_ref)).unsqueeze(0)

        # rest-frame sobre el cubo CRUDO, gated por la certeza del canal v
        if self.restframe:
            cube = restframe_shift(cube, maps[_VSTAR_CH], c_phys[_VSTAR_CH],
                                   self.dloglam)

        # peso fisico: masa RAW (sin z-score) >= 0, enmascarada (C7)
        w_phys_mass = torch.nan_to_num(maps[_MASS_CH], 0.0).clamp_min(0.0) * M

        # normalizacion
        cube = _zscore(torch.from_numpy(symlog(cube.numpy())),
                       self._cube_mean.view(-1, 1, 1),
                       self._cube_std.view(-1, 1, 1))
        image = _zscore(image, self._img_mean, self._img_std)
        maps = _zscore(maps, self._map_mean, self._map_std)

        # c=0 exactamente donde ~M_valid o NaN original; nan_to_num despues
        # (el valor da igual: c=0 lo anula, specs/45 P1)
        for c in (c_phys, c_spat, c_spec):
            c[..., ~M] = 0.0
        c_phys[torch.isnan(maps)] = 0.0
        c_spat[torch.isnan(image)] = 0.0
        cube = torch.nan_to_num(cube, 0.0)
        image = torch.nan_to_num(image, 0.0)
        maps = torch.nan_to_num(maps, 0.0)

        sample: dict = {
            "cube": cube, "c_spec": c_spec,
            "image": image, "c_spat": c_spat,
            "maps": maps, "c_phys": c_phys,
            "psf": psf,
            **labels,
            "M": M, "M_unc_mass": M_unc_mass, "M_unc_lum": M_unc_lum,
            "w_phys_mass": w_phys_mass,
            "target_fractions_mass": tf_mass, "target_fractions_lum": tf_lum,
            "galaxy_id": gid, "view_id": view,
        }

        if self._augment_pipe is not None:
            rng = np.random.default_rng(np.array(
                [self.seed, self._epoch, idx], dtype=np.uint64))
            sample = self._augment_pipe(sample, rng)

        if self.mult > 1:
            sample = pad_to_multiple(sample, self.mult)
        else:
            sample["hw_native"] = tuple(M.shape)
        return sample
