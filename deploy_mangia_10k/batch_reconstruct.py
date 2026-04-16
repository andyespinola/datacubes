import argparse
import csv
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter

from rss_to_cube_mangia_official import (
    DEFAULT_CATALOG,
    DEFAULT_TEMPLATE,
    infer_n_fib,
    lookup_reff_from_catalog,
    run_official_regrid,
    strip_rss_suffix,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruccion batch de cubos MaNGIA oficiales a partir de RSS"
    )
    parser.add_argument("--rss-dir", required=True, help="Directorio raiz con archivos RSS")
    parser.add_argument("--output-dir", required=True, help="Directorio de salida para los cubos")
    parser.add_argument(
        "--catalog",
        default=str(DEFAULT_CATALOG),
        help="Catalogo MaNGIA para resolver re_kpc automaticamente",
    )
    parser.add_argument(
        "--template-ssp-control",
        default=str(DEFAULT_TEMPLATE),
        help="Template SSP de control requerido por el codigo oficial",
    )
    parser.add_argument(
        "--rss-glob",
        default="*.cube_RSS.fits*",
        help="Patron glob recursivo para encontrar RSS",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Cantidad de procesos paralelos. La reconstruccion actual usa CPU, no GPU.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Indice inicial dentro de la lista ordenada de RSS",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Cantidad maxima de RSS a procesar desde start-index. 0 significa todos.",
    )
    parser.add_argument(
        "--include-gas",
        action="store_true",
        help="Suma la componente gaseosa al cubo principal",
    )
    parser.add_argument("--thet", type=float, default=0.0, help="Offset angular en grados")
    parser.add_argument(
        "--r-eff",
        type=float,
        default=None,
        help="Override global de R_eff en kpc. Si se omite, se resuelve desde el catalogo.",
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocesa incluso si ya existen .cube.fits.gz y .cube_val.fits.gz",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Archivo CSV de log. Si se omite, se crea uno dentro de logs/",
    )
    return parser.parse_args()


def expected_outputs(out_prefix):
    out_prefix = Path(out_prefix)
    cube_path = out_prefix.parent / f"{out_prefix.name}.cube.fits.gz"
    cube_val_path = out_prefix.parent / f"{out_prefix.name}.cube_val.fits.gz"
    return cube_path, cube_val_path


def discover_rss_files(rss_dir, pattern):
    rss_dir = Path(rss_dir).resolve()
    if not rss_dir.exists():
        raise FileNotFoundError(f"No existe RSS_DIR: {rss_dir}")

    files = [path for path in rss_dir.rglob(pattern) if path.is_file()]
    files.sort()
    return files


def select_worklist(files, start_index, count):
    if start_index < 0:
        raise ValueError("start-index no puede ser negativo")
    if count < 0:
        raise ValueError("count no puede ser negativo")

    sliced = files[start_index:]
    if count > 0:
        sliced = sliced[:count]
    return sliced


def make_task(
    rss_path,
    output_dir,
    template_ssp_control,
    catalog,
    include_gas,
    thet,
    r_eff,
    noise_sn,
    noise_radius,
    force,
):
    return {
        "rss_path": str(Path(rss_path).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "template_ssp_control": str(Path(template_ssp_control).resolve()),
        "catalog": str(Path(catalog).resolve()),
        "include_gas": bool(include_gas),
        "thet": float(thet),
        "r_eff": None if r_eff is None else float(r_eff),
        "noise_sn": float(noise_sn),
        "noise_radius": float(noise_radius),
        "force": bool(force),
    }


def process_one(task):
    rss_path = Path(task["rss_path"]).resolve()
    output_dir = Path(task["output_dir"]).resolve()
    out_prefix = output_dir / strip_rss_suffix(rss_path)
    cube_path, cube_val_path = expected_outputs(out_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    t0 = perf_counter()

    try:
        if not task["force"] and cube_path.exists() and cube_val_path.exists():
            elapsed = perf_counter() - t0
            return {
                "status": "skipped",
                "rss_path": str(rss_path),
                "out_prefix": str(out_prefix),
                "cube_path": str(cube_path),
                "cube_val_path": str(cube_val_path),
                "n_fib": "",
                "r_eff": "",
                "seconds": f"{elapsed:.2f}",
                "started_at": started_at,
                "finished_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "error": "",
            }

        n_fib = infer_n_fib(rss_path)
        if task["r_eff"] is None:
            r_eff, _ids = lookup_reff_from_catalog(rss_path, task["catalog"])
        else:
            r_eff = task["r_eff"]

        run_official_regrid(
            rss_fits=rss_path,
            out_prefix=out_prefix,
            template_ssp_control=task["template_ssp_control"],
            n_fib=n_fib,
            thet=task["thet"],
            r_eff=r_eff,
            include_gas=task["include_gas"],
            noise_sn=task["noise_sn"],
            noise_radius=task["noise_radius"],
        )

        elapsed = perf_counter() - t0
        return {
            "status": "ok",
            "rss_path": str(rss_path),
            "out_prefix": str(out_prefix),
            "cube_path": str(cube_path),
            "cube_val_path": str(cube_val_path),
            "n_fib": str(n_fib),
            "r_eff": f"{r_eff:.8f}",
            "seconds": f"{elapsed:.2f}",
            "started_at": started_at,
            "finished_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "error": "",
        }
    except Exception as exc:
        elapsed = perf_counter() - t0
        return {
            "status": "error",
            "rss_path": str(rss_path),
            "out_prefix": str(out_prefix),
            "cube_path": str(cube_path),
            "cube_val_path": str(cube_val_path),
            "n_fib": "",
            "r_eff": "",
            "seconds": f"{elapsed:.2f}",
            "started_at": started_at,
            "finished_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "error": f"{exc}\n{traceback.format_exc()}",
        }


def default_log_path():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs") / f"batch_reconstruct_{stamp}.csv"


def write_header_if_needed(writer, file_exists):
    if file_exists:
        return
    writer.writerow(
        [
            "status",
            "rss_path",
            "out_prefix",
            "cube_path",
            "cube_val_path",
            "n_fib",
            "r_eff",
            "seconds",
            "started_at",
            "finished_at",
            "error",
        ]
    )


def main():
    args = parse_args()

    rss_files = discover_rss_files(args.rss_dir, args.rss_glob)
    selected = select_worklist(rss_files, args.start_index, args.count)

    if not selected:
        print("No se encontraron RSS para procesar con los filtros dados.")
        return

    log_path = Path(args.log_file) if args.log_file else default_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"RSS encontrados: {len(rss_files)}")
    print(f"RSS seleccionados: {len(selected)}")
    print(f"Workers: {args.workers}")
    print(f"Output dir: {Path(args.output_dir).resolve()}")
    print(f"Log file: {log_path.resolve()}")
    print("Nota: la reconstruccion oficial actual usa CPU; la GPU no se aprovecha en este pipeline.")

    tasks = [
        make_task(
            rss_path=path,
            output_dir=args.output_dir,
            template_ssp_control=args.template_ssp_control,
            catalog=args.catalog,
            include_gas=args.include_gas,
            thet=args.thet,
            r_eff=args.r_eff,
            noise_sn=args.noise_sn,
            noise_radius=args.noise_radius,
            force=args.force,
        )
        for path in selected
    ]

    completed = 0
    ok_count = 0
    skipped_count = 0
    error_count = 0

    file_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        write_header_if_needed(writer, file_exists)

        if args.workers <= 1:
            for index, task in enumerate(tasks, start=1):
                result = process_one(task)
                writer.writerow(
                    [
                        result["status"],
                        result["rss_path"],
                        result["out_prefix"],
                        result["cube_path"],
                        result["cube_val_path"],
                        result["n_fib"],
                        result["r_eff"],
                        result["seconds"],
                        result["started_at"],
                        result["finished_at"],
                        result["error"],
                    ]
                )
                handle.flush()
                completed += 1
                if result["status"] == "ok":
                    ok_count += 1
                elif result["status"] == "skipped":
                    skipped_count += 1
                else:
                    error_count += 1
                print(
                    f"[{completed}/{len(tasks)}] {result['status'].upper()} :: "
                    f"{Path(result['rss_path']).name} :: {result['seconds']} s"
                )
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(process_one, task) for task in tasks]
                for future in as_completed(futures):
                    result = future.result()
                    writer.writerow(
                        [
                            result["status"],
                            result["rss_path"],
                            result["out_prefix"],
                            result["cube_path"],
                            result["cube_val_path"],
                            result["n_fib"],
                            result["r_eff"],
                            result["seconds"],
                            result["started_at"],
                            result["finished_at"],
                            result["error"],
                        ]
                    )
                    handle.flush()
                    completed += 1
                    if result["status"] == "ok":
                        ok_count += 1
                    elif result["status"] == "skipped":
                        skipped_count += 1
                    else:
                        error_count += 1
                    print(
                        f"[{completed}/{len(tasks)}] {result['status'].upper()} :: "
                        f"{Path(result['rss_path']).name} :: {result['seconds']} s"
                    )

    print(
        "Resumen final :: "
        f"ok={ok_count}, skipped={skipped_count}, error={error_count}, total={len(tasks)}"
    )


if __name__ == "__main__":
    main()
