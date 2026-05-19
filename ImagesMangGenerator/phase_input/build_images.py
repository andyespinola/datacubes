from __future__ import annotations

import argparse
import json
from pathlib import Path

from .image_provider import (
    CatalogImageBuilder,
    ImageProvider,
    ImageProviderConfig,
    ImageProviderInput,
    bootstrap_pilot_data,
    discover_mangia_cubes,
    output_path_for,
    save_provided_image,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build g/r/i image tensors for MaNGIA and MaNGA cubes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap-pilot", help="Copy pilot FITS files from the legacy cubes workspace.")
    bootstrap.add_argument("--source-root", default="/home/andy/pythonprojects/cubes")
    bootstrap.add_argument("--destination-root", default="ImagesMangGenerator/data/pilot")

    one_mangia = subparsers.add_parser("one-mangia", help="Synthesize one MaNGIA image from one cube.")
    one_mangia.add_argument("--cube", required=True)
    one_mangia.add_argument("--galaxy-id")
    one_mangia.add_argument("--view-id", type=int)
    one_mangia.add_argument("--outdir", required=True)
    add_common_config_args(one_mangia)

    catalog_mangia = subparsers.add_parser("catalog-mangia", help="Synthesize all MaNGIA FITS cubes in a directory.")
    catalog_mangia.add_argument("--mangia-root", "--input-dir", dest="mangia_root", required=True)
    catalog_mangia.add_argument("--outdir", "--output-dir", dest="outdir", required=True)
    catalog_mangia.add_argument("--manifest")
    catalog_mangia.add_argument("--workers", type=int, default=1)
    catalog_mangia.add_argument("--pattern", default="*.cube.fits.gz", help="Input filename glob.")
    catalog_mangia.add_argument("--recursive", action="store_true", help="Search input directory recursively.")
    catalog_mangia.add_argument("--limit", type=int, help="Process only the first N discovered cubes.")
    catalog_mangia.add_argument("--skip-existing", action="store_true", help="Do not rebuild NPZ files that already exist.")
    catalog_mangia.add_argument("--dry-run", action="store_true", help="Only list discovered inputs.")
    add_common_config_args(catalog_mangia)

    one_manga = subparsers.add_parser("one-manga", help="Build one MaNGA image from SDSS cutouts.")
    one_manga.add_argument("--cube", required=True)
    one_manga.add_argument("--plateifu", required=True)
    one_manga.add_argument("--ra", type=float, required=True)
    one_manga.add_argument("--dec", type=float, required=True)
    one_manga.add_argument("--ifusize", type=int, default=127)
    one_manga.add_argument("--outdir", required=True)
    one_manga.add_argument("--cache-dir", default="ImagesMangGenerator/data/cache/sdss")
    add_common_config_args(one_manga)

    catalog_manga = subparsers.add_parser("catalog-manga", help="Build MaNGA images from drpall and local cubes.")
    catalog_manga.add_argument("--drpall", required=True)
    catalog_manga.add_argument("--cubes-dir", required=True)
    catalog_manga.add_argument("--outdir", required=True)
    catalog_manga.add_argument("--cache-dir")
    catalog_manga.add_argument("--manifest")
    catalog_manga.add_argument("--workers", type=int, default=1)
    catalog_manga.add_argument("--limit", type=int)
    add_common_config_args(catalog_manga)

    return parser


def add_common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-shape",
        default=None,
        help="Shape as H,W, or 'native' to preserve each cube's spatial grid.",
    )
    parser.add_argument("--output-unit", default="nanomaggie", choices=["nanomaggie", "ab_flux", "ab_mag_arcsec2"])
    parser.add_argument("--on-missing-band", default="fail", choices=["skip", "interpolate", "fail"])
    parser.add_argument("--on-wcs-failure", default="fallback_synthesis", choices=["fallback_synthesis", "fail"])
    parser.add_argument("--sdss-source", default="astroquery", choices=["astroquery", "skyserver", "sas"])
    parser.add_argument("--sdss-cutout-size-arcsec", type=float, default=80.0)


def config_from_args(args: argparse.Namespace, default_shape: tuple[int, int] | None) -> ImageProviderConfig:
    output_shape = default_shape
    if args.output_shape:
        if args.output_shape.lower() == "native":
            output_shape = None
        else:
            parts = [int(part.strip()) for part in args.output_shape.split(",")]
            if len(parts) != 2:
                raise ValueError("--output-shape must be formatted as H,W or 'native'")
            output_shape = (parts[0], parts[1])
    return ImageProviderConfig(
        output_shape=output_shape,
        output_unit=args.output_unit,
        on_missing_band=args.on_missing_band,
        on_wcs_failure=args.on_wcs_failure,
        sdss_source=args.sdss_source,
        sdss_cutout_size_arcsec=args.sdss_cutout_size_arcsec,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "bootstrap-pilot":
        copied = bootstrap_pilot_data(args.source_root, args.destination_root)
        print(json.dumps({"copied": [str(path) for path in copied]}, indent=2))
        return

    provider = ImageProvider()
    builder = CatalogImageBuilder(provider)

    if args.command == "one-mangia":
        cube = Path(args.cube)
        galaxy_id = args.galaxy_id or cube.name.replace(".cube.fits.gz", "").replace(".fits.gz", "")
        config = config_from_args(args, None)
        provided = provider.provide(
            ImageProviderInput(
                mode="mangia",
                galaxy_id=galaxy_id,
                view_id=args.view_id,
                cube_path=cube,
                config=config,
            )
        )
        output_path = save_provided_image(provided, output_path_for(args.outdir, provided))
        print(json.dumps({"output_path": str(output_path), "metadata": provided.metadata()}, indent=2))
        return

    if args.command == "catalog-mangia":
        if args.dry_run:
            entries = discover_mangia_cubes(args.mangia_root, pattern=args.pattern, recursive=args.recursive)
            if args.limit is not None:
                entries = entries[: args.limit]
            print(json.dumps({"total": len(entries), "inputs": [str(path) for path in entries]}, indent=2))
            return

        config = config_from_args(args, None)
        report = builder.build_mangia_catalog(
            args.mangia_root,
            args.outdir,
            config,
            n_workers=args.workers,
            manifest_path=args.manifest,
            pattern=args.pattern,
            recursive=args.recursive,
            limit=args.limit,
            skip_existing=args.skip_existing,
        )
        print(json.dumps(report.summary(), indent=2))
        return

    if args.command == "one-manga":
        config = config_from_args(args, (74, 74))
        row = {
            "plateifu": args.plateifu,
            "objra": args.ra,
            "objdec": args.dec,
            "ifudesignsize": args.ifusize,
        }
        provided = provider.provide(
            ImageProviderInput(
                mode="manga",
                galaxy_id=args.plateifu,
                cube_path=Path(args.cube),
                drpall_row=row,
                cache_dir=Path(args.cache_dir),
                config=config,
            )
        )
        output_path = save_provided_image(provided, output_path_for(args.outdir, provided))
        print(json.dumps({"output_path": str(output_path), "metadata": provided.metadata()}, indent=2))
        return

    if args.command == "catalog-manga":
        config = config_from_args(args, (74, 74))
        report = builder.build_manga_catalog(
            args.drpall,
            args.cubes_dir,
            args.outdir,
            config,
            n_workers=args.workers,
            cache_dir=args.cache_dir,
            manifest_path=args.manifest,
            limit=args.limit,
        )
        print(json.dumps(report.summary(), indent=2))
        return


if __name__ == "__main__":
    main()
