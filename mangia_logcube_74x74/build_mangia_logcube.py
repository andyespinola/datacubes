import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from harmonize_logcube import harmonize_official_cube, load_config
from validate_logcube import validate_product


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import rss_to_cube_mangia_official as mangia_official


DEFAULT_CONFIG = PROJECT_DIR / "default_config.json"


def build_parser():
    parser = argparse.ArgumentParser(description="Construye un cubo MaNGIA armonizado a LOGCUBE-like 74x74")
    parser.add_argument("rss_fits", help="Ruta al RSS de MaNGIA")
    parser.add_argument(
        "--reference-logcube",
        default=None,
        help="LOGCUBE real usado como template. Si se omite, se usa el default del config.",
    )
    parser.add_argument(
        "--outdir",
        default=str(PROJECT_DIR / "output"),
        help="Directorio de salida",
    )
    parser.add_argument(
        "--keep-official",
        action="store_true",
        help="Conservar también el cubo oficial en el directorio de salida",
    )
    parser.add_argument(
        "--include-gas",
        action="store_true",
        help="Pedir al flujo oficial que incluya gas en el PRIMARY",
    )
    parser.add_argument(
        "--catalog",
        default=str(mangia_official.DEFAULT_CATALOG),
        help="Catálogo MaNGIA para resolver re_kpc",
    )
    parser.add_argument(
        "--template-ssp-control",
        default=str(mangia_official.DEFAULT_TEMPLATE),
        help="Template SSP requerido por el flujo oficial",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Archivo JSON de configuración del proyecto",
    )
    parser.add_argument("--n-fib", type=int, default=None, help="Override manual de n_fib")
    parser.add_argument("--r-eff", type=float, default=None, help="Override manual de re_kpc")
    parser.add_argument("--thet", type=float, default=0.0, help="Offset angular en grados")
    parser.add_argument("--noise-sn", type=float, default=5.0, help="S/N objetivo del upstream")
    parser.add_argument("--noise-radius", type=float, default=2.0, help="Radio en unidades de Re para el upstream")
    return parser


def resolve_reference_logcube(config, provided):
    if provided:
        return Path(provided).resolve()
    default_reference = config.get("default_reference_logcube")
    if not default_reference:
        raise ValueError("No definiste --reference-logcube ni hay default_reference_logcube en el config")
    return (PROJECT_DIR / default_reference).resolve()


def maybe_copy(src, dst):
    src = Path(src).resolve()
    dst = Path(dst).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    rss_fits = Path(args.rss_fits).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    reference_logcube = resolve_reference_logcube(config, args.reference_logcube)

    if not rss_fits.exists():
        raise FileNotFoundError(f"No existe el RSS: {rss_fits}")
    if not reference_logcube.exists():
        raise FileNotFoundError(f"No existe el LOGCUBE de referencia: {reference_logcube}")

    out_prefix_name = mangia_official.strip_rss_suffix(rss_fits)
    final_product_path = outdir / f"{out_prefix_name}.manga_logcube_74x74.fits.gz"
    summary_path = outdir / f"{out_prefix_name}.manga_logcube_74x74.summary.json"

    n_fib = args.n_fib if args.n_fib is not None else mangia_official.infer_n_fib(rss_fits)
    if args.r_eff is not None:
        r_eff = args.r_eff
        ids = mangia_official.parse_mangia_rss_name(rss_fits)
    else:
        r_eff, ids = mangia_official.lookup_reff_from_catalog(rss_fits, args.catalog)

    with tempfile.TemporaryDirectory(prefix="mangia_logcube74_") as temp_dir:
        temp_dir = Path(temp_dir)
        official_prefix = temp_dir / out_prefix_name
        mangia_official.run_official_regrid(
            rss_fits=rss_fits,
            out_prefix=official_prefix,
            template_ssp_control=args.template_ssp_control,
            n_fib=n_fib,
            thet=args.thet,
            r_eff=r_eff,
            include_gas=args.include_gas,
            noise_sn=args.noise_sn,
            noise_radius=args.noise_radius,
        )

        official_cube = official_prefix.with_suffix(".cube.fits.gz")
        official_cube_val = official_prefix.with_suffix(".cube_val.fits.gz")
        if not official_cube.exists():
            raise FileNotFoundError(f"El flujo oficial no produjo el cubo esperado: {official_cube}")

        if args.keep_official:
            kept_official_cube = outdir / official_cube.name
            kept_official_cube_val = outdir / official_cube_val.name if official_cube_val.exists() else None
            maybe_copy(official_cube, kept_official_cube)
            if kept_official_cube_val is not None:
                maybe_copy(official_cube_val, kept_official_cube_val)
            official_for_validation = kept_official_cube
        else:
            kept_official_cube = None
            kept_official_cube_val = None
            official_for_validation = official_cube

        harmonize_result = harmonize_official_cube(
            official_cube=official_cube,
            reference_logcube=reference_logcube,
            output_path=final_product_path,
            config=config,
        )
        validation = validate_product(
            product_path=final_product_path,
            reference_logcube=reference_logcube,
            official_cube=official_for_validation,
            config=config,
        )
        if not args.keep_official:
            validation["official_cube"] = None
            validation["official_cube_temp_used"] = True

    summary = {
        "rss_fits": str(rss_fits),
        "canonical_id": f"TNG50-{ids['snapshot']}-{ids['subhalo_id']}-{ids['view']}-{ids['ifu']}",
        "reference_logcube": str(reference_logcube),
        "config_path": str(Path(args.config).resolve()),
        "n_fib": int(n_fib),
        "r_eff_kpc": float(r_eff),
        "include_gas": bool(args.include_gas),
        "official_cube_kept": None if kept_official_cube is None else str(kept_official_cube),
        "official_cube_val_kept": None if kept_official_cube_val is None else str(kept_official_cube_val),
        "harmonized_product": str(final_product_path),
        "harmonization": harmonize_result,
        "validation": validation,
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if validation["all_checks_passed"] else 1)


if __name__ == "__main__":
    main()
