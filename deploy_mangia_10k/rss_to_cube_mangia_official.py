import argparse
import importlib.util
import os
import re
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from astropy.io import fits


UPSTREAM_REPO = "https://github.com/illustristng/MaNGIA_TNG"
UPSTREAM_COMMIT = "d859f3e24704f983d0eabfc080daf346a6d7b5da"

BASE_DIR = Path(__file__).resolve().parent
OFFICIAL_DIR = BASE_DIR / "official_mangia"
OFFICIAL_SOURCE = OFFICIAL_DIR / "sin_ifu_clean.py"
DEFAULT_TEMPLATE = OFFICIAL_DIR / "libs" / "MaStar_CB19.slog_1_5.fits.gz"
DEFAULT_CATALOG = BASE_DIR / "MaNGIA_catalog.fits"

FIBER_COUNT_TO_N_FIB = {
    19: 3,
    37: 4,
    61: 5,
    91: 6,
    127: 7,
}


@contextmanager
def pushd(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def strip_rss_suffix(path):
    name = path.name
    if name.endswith(".fits.gz"):
        name = name[:-8]
    elif name.endswith(".fits"):
        name = name[:-5]

    if name.endswith(".cube_RSS"):
        name = name[: -len(".cube_RSS")]

    return name


def parse_mangia_rss_name(rss_path):
    name = Path(rss_path).name
    pattern = re.compile(
        r"^TNG50-(?P<snapshot>\d+)-(?P<subhalo_id>\d+)-(?P<view>\d+)-(?P<ifu>\d+)(?:\.[^.]+)?\.cube_RSS(?:\.fits(?:\.gz)?)?$"
    )
    match = pattern.match(name)
    if match is None:
        raise ValueError(
            "No pude extraer snapshot/subhalo_id/view desde el nombre del RSS. "
            "Esperaba un patrón tipo TNG50-87-141934-0-127.cube_RSS.fits"
        )

    return {
        "snapshot": int(match.group("snapshot")),
        "subhalo_id": int(match.group("subhalo_id")),
        "view": int(match.group("view")),
        "ifu": int(match.group("ifu")),
    }


def infer_n_fib(rss_path):
    with fits.open(rss_path) as hdul:
        raw_value = str(hdul[0].header.get("IFUCON", "")).strip()
        if raw_value:
            try:
                total_fibers = int(raw_value)
            except ValueError:
                total_fibers = None
            else:
                if total_fibers in FIBER_COUNT_TO_N_FIB:
                    return FIBER_COUNT_TO_N_FIB[total_fibers]

    raise ValueError(
        "No pude inferir n_fib desde IFUCON. "
        "Pásalo explícitamente con --n-fib."
    )


def lookup_reff_from_catalog(rss_path, catalog_path):
    catalog_path = Path(catalog_path).resolve()
    if not catalog_path.exists():
        raise FileNotFoundError(f"No existe el catálogo MaNGIA: {catalog_path}")

    ids = parse_mangia_rss_name(rss_path)
    with fits.open(catalog_path) as hdul:
        if len(hdul) < 2 or hdul[1].data is None:
            raise ValueError("El catálogo MaNGIA no tiene una tabla binaria utilizable en HDU[1]")

        data = hdul[1].data
        required = {"snapshot", "subhalo_id", "view", "re_kpc"}
        names = set(data.names or [])
        missing = required - names
        if missing:
            raise ValueError(
                "Faltan columnas requeridas en el catálogo MaNGIA: "
                + ", ".join(sorted(missing))
            )

        rows = data[
            (data["snapshot"] == ids["snapshot"])
            & (data["subhalo_id"] == ids["subhalo_id"])
            & (data["view"] == ids["view"])
        ]

        if len(rows) == 0:
            raise LookupError(
                "No encontré una fila en el catálogo MaNGIA para "
                f"snapshot={ids['snapshot']}, subhalo_id={ids['subhalo_id']}, view={ids['view']}"
            )
        if len(rows) > 1:
            raise LookupError(
                "Encontré múltiples filas en el catálogo MaNGIA para "
                f"snapshot={ids['snapshot']}, subhalo_id={ids['subhalo_id']}, view={ids['view']}"
            )

        row = rows[0]
        re_kpc = float(row["re_kpc"])
        if not np.isfinite(re_kpc) or re_kpc <= 0:
            raise ValueError(f"re_kpc inválido en catálogo: {re_kpc}")

        return re_kpc, ids


def ensure_stub_illustris_python():
    if "illustris_python" not in sys.modules:
        sys.modules["illustris_python"] = types.ModuleType("illustris_python")


def load_official_module():
    if not OFFICIAL_SOURCE.exists():
        raise FileNotFoundError(
            f"No existe la fuente oficial local: {OFFICIAL_SOURCE}"
        )

    ensure_stub_illustris_python()

    spec = importlib.util.spec_from_file_location(
        "mangia_official_sin_ifu_clean",
        OFFICIAL_SOURCE,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    # El upstream usa np.np.float32 en una rama; este alias lo vuelve inocuo.
    if not hasattr(module.np, "np"):
        module.np.np = module.np

    return module


def run_official_regrid(
    rss_fits,
    out_prefix,
    template_ssp_control,
    n_fib,
    thet,
    r_eff,
    include_gas,
    noise_sn,
    noise_radius,
):
    rss_fits = Path(rss_fits).resolve()
    out_prefix = Path(out_prefix).resolve()
    template_ssp_control = Path(template_ssp_control).resolve()

    if not rss_fits.exists():
        raise FileNotFoundError(f"No existe el RSS FITS: {rss_fits}")
    if not template_ssp_control.exists():
        raise FileNotFoundError(f"No existe el template SSP: {template_ssp_control}")

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    official = load_official_module()

    with pushd(OFFICIAL_DIR):
        official.regrid(
            rss_file=rss_fits.name,
            outf=out_prefix.name,
            template_SSP_control=str(template_ssp_control),
            dir_r=str(rss_fits.parent) + os.sep,
            dir_o=str(out_prefix.parent) + os.sep,
            n_fib=n_fib,
            thet=thet,
            R_eff=r_eff,
            include_gas=include_gas,
            noise=[noise_sn, noise_radius],
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Wrapper local para la reconstrucción oficial MaNGIA (regrid)"
    )
    parser.add_argument("rss_fits", help="Ruta al archivo RSS FITS")
    parser.add_argument(
        "--out-prefix",
        default=None,
        help=(
            "Prefijo de salida. El código oficial escribirá "
            "<prefijo>.cube.fits.gz y <prefijo>.cube_val.fits.gz"
        ),
    )
    parser.add_argument(
        "--template-ssp-control",
        default=str(DEFAULT_TEMPLATE),
        help="Template SSP de control requerido por el código oficial",
    )
    parser.add_argument(
        "--n-fib",
        type=int,
        default=None,
        help="Radio del bundle IFU en fibras (3..7). Si se omite, se infiere desde IFUCON.",
    )
    parser.add_argument("--thet", type=float, default=0.0, help="Offset angular en grados")
    parser.add_argument(
        "--r-eff",
        type=float,
        default=None,
        help=(
            "Radio efectivo en kpc. Si se omite, se busca automáticamente "
            "en MaNGIA_catalog.fits a partir del nombre del RSS."
        ),
    )
    parser.add_argument(
        "--catalog",
        default=str(DEFAULT_CATALOG),
        help="Catálogo MaNGIA usado para resolver re_kpc automáticamente",
    )
    parser.add_argument(
        "--include-gas",
        action="store_true",
        help="Incluir emisión gaseosa en el cubo final",
    )
    parser.add_argument(
        "--noise-sn",
        type=float,
        default=5.0,
        help="S/N objetivo usado por el upstream",
    )
    parser.add_argument(
        "--noise-radius",
        type=float,
        default=2.0,
        help="Radio en unidades de Re donde se referencia el S/N",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    rss_fits = Path(args.rss_fits).resolve()
    if args.out_prefix is None:
        out_prefix = rss_fits.with_name(strip_rss_suffix(rss_fits))
    else:
        out_prefix = Path(args.out_prefix).resolve()

    n_fib = args.n_fib if args.n_fib is not None else infer_n_fib(rss_fits)
    if args.r_eff is not None:
        r_eff = args.r_eff
        source = "manual"
        ids = None
    else:
        r_eff, ids = lookup_reff_from_catalog(rss_fits, args.catalog)
        source = f"catalog: {Path(args.catalog).resolve()}"

    print(f"Usando MaNGIA oficial desde {UPSTREAM_REPO} @ {UPSTREAM_COMMIT}")
    print(f"RSS: {rss_fits}")
    print(f"Output prefix: {out_prefix}")
    print(f"Template SSP control: {Path(args.template_ssp_control).resolve()}")
    print(f"n_fib: {n_fib}")
    if ids is not None:
        print(
            "Catalog match: "
            f"snapshot={ids['snapshot']}, subhalo_id={ids['subhalo_id']}, view={ids['view']}"
        )
    print(f"R_eff [kpc]: {r_eff} ({source})")

    run_official_regrid(
        rss_fits=rss_fits,
        out_prefix=out_prefix,
        template_ssp_control=args.template_ssp_control,
        n_fib=n_fib,
        thet=args.thet,
        r_eff=r_eff,
        include_gas=args.include_gas,
        noise_sn=args.noise_sn,
        noise_radius=args.noise_radius,
    )


if __name__ == "__main__":
    main()
