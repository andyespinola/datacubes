from __future__ import annotations

import csv
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from .io import (
    output_paths,
    read_mangia_official_cube,
    read_manifest_cube_paths,
    resolve_template_path,
    unique_paths,
    write_fits,
    write_npz,
)
from .models import KinematicMaps, KinematicMomentsConfig, OfficialCube
from .ppxf_fit import build_fit_grid, compute_snr, fit_spaxel, load_mastar_templates


def append_run_log(log_path: str | Path | None, message: str) -> None:
    if log_path is None:
        return
    path = Path(log_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {message}\n")


def _empty_maps(cube: OfficialCube, config: KinematicMomentsConfig) -> KinematicMaps:
    ny, nx = cube.spatial_shape
    nan_map = lambda: np.full((ny, nx), np.nan, dtype=np.float32)
    coverage = np.any(cube.valid_cube, axis=0).astype(np.uint8)
    return KinematicMaps(
        galaxy_id=cube.galaxy_id,
        cube_path=cube.path,
        h3=nan_map(),
        h4=nan_map(),
        v_ppxf=nan_map(),
        sigma_ppxf=nan_map(),
        h3_err=nan_map(),
        h4_err=nan_map(),
        quality_mask=np.zeros((ny, nx), dtype=np.uint8),
        coverage_mask=coverage,
        snr_map=nan_map(),
        chi2_map=nan_map(),
        n_spaxels_fitted=0,
        n_quality_ok=0,
        config_summary=_config_summary(config),
    )


def _config_summary(config: KinematicMomentsConfig) -> dict[str, object]:
    summary = asdict(config)
    if summary.get("template_path") is not None:
        summary["template_path"] = str(summary["template_path"])
    return summary


def _spaxel_order(coverage: np.ndarray) -> list[tuple[int, int]]:
    y, x = np.nonzero(coverage)
    cy = (coverage.shape[0] - 1) / 2.0
    cx = (coverage.shape[1] - 1) / 2.0
    order = np.argsort((y - cy) ** 2 + (x - cx) ** 2)
    return [(int(y[idx]), int(x[idx])) for idx in order]


def extract_kinematics(
    cube: OfficialCube,
    config: KinematicMomentsConfig,
    max_spaxels: int | None = None,
    show_progress: bool = True,
) -> KinematicMaps:
    template_path = resolve_template_path(config.template_path)
    config = KinematicMomentsConfig(**{**asdict(config), "template_path": template_path})
    grid = build_fit_grid(cube.wave, cube.redshift, config)
    templates = load_mastar_templates(template_path, grid.velscale, config)
    maps = _empty_maps(cube, config)
    wave_rest = cube.wave / (1.0 + cube.redshift)

    positions = _spaxel_order(maps.coverage_mask > 0)
    if max_spaxels is not None:
        positions = positions[: int(max_spaxels)]

    failures = 0
    iterator = tqdm(positions, desc=cube.galaxy_id, disable=not show_progress)
    for y, x in iterator:
        valid = cube.valid_cube[:, y, x]
        snr = compute_snr(cube.flux[:, y, x], cube.error[:, y, x], wave_rest, valid, config)
        maps.snr_map[y, x] = snr
        if not np.isfinite(snr) or snr < config.snr_min:
            continue
        try:
            v, sigma, h3, h4, h3_err, h4_err, chi2 = fit_spaxel(
                cube.flux[:, y, x],
                cube.error[:, y, x],
                valid,
                grid,
                templates,
                config,
            )
        except Exception:
            failures += 1
            continue

        maps.v_ppxf[y, x] = v
        maps.sigma_ppxf[y, x] = sigma
        maps.h3[y, x] = h3
        maps.h4[y, x] = h4
        maps.h3_err[y, x] = h3_err
        maps.h4_err[y, x] = h4_err
        maps.chi2_map[y, x] = chi2
        maps.quality_mask[y, x] = 1
        maps.n_spaxels_fitted += 1

    maps.n_quality_ok = int(np.sum(maps.quality_mask > 0))
    if failures:
        maps.message = f"{failures} spaxels failed pPXF"
    return maps


def process_cube(
    cube_path: str | Path,
    output_dir: str | Path,
    config: KinematicMomentsConfig,
    max_spaxels: int | None = None,
    overwrite: bool = False,
    show_progress: bool = True,
    log_path: str | Path | None = None,
) -> dict[str, object]:
    cube_path = Path(cube_path).expanduser().resolve()
    npz_path, fits_path = output_paths(cube_path, output_dir)
    append_run_log(log_path, f"START cube={cube_path}")
    if not overwrite and npz_path.exists() and fits_path.exists():
        append_run_log(
            log_path,
            f"SKIP cube={cube_path} reason='outputs already exist' npz={npz_path} fits={fits_path}",
        )
        return {
            "cube_path": str(cube_path),
            "status": "skipped",
            "n_spaxels_fitted": "",
            "n_quality_ok": "",
            "snr_median": "",
            "chi2_median": "",
            "npz_path": str(npz_path),
            "fits_path": str(fits_path),
            "message": "outputs already exist",
        }

    try:
        cube = read_mangia_official_cube(cube_path)
        rest_wave = cube.wave / (1.0 + cube.redshift)
        append_run_log(
            log_path,
            "CUBE "
            f"path={cube_path} shape={cube.flux.shape} redshift={cube.redshift} "
            f"wave_range=({float(np.nanmin(cube.wave))}, {float(np.nanmax(cube.wave))}) "
            f"rest_range=({float(np.nanmin(rest_wave))}, {float(np.nanmax(rest_wave))}) "
            f"coverage_spaxels={int(np.any(cube.valid_cube, axis=0).sum())}",
        )
        maps = extract_kinematics(cube, config, max_spaxels=max_spaxels, show_progress=show_progress)
        write_npz(maps, npz_path)
        write_fits(maps, cube.header, fits_path)

        quality = maps.quality_mask > 0
        snr_median = float(np.nanmedian(maps.snr_map[quality])) if np.any(quality) else float("nan")
        chi2_median = float(np.nanmedian(maps.chi2_map[quality])) if np.any(quality) else float("nan")
        append_run_log(
            log_path,
            "OK "
            f"cube={cube_path} n_spaxels_fitted={maps.n_spaxels_fitted} "
            f"n_quality_ok={maps.n_quality_ok} snr_median={snr_median} "
            f"chi2_median={chi2_median} npz={npz_path} fits={fits_path} "
            f"message={maps.message!r}",
        )
        return {
            "cube_path": str(cube_path),
            "status": "ok",
            "n_spaxels_fitted": maps.n_spaxels_fitted,
            "n_quality_ok": maps.n_quality_ok,
            "snr_median": snr_median,
            "chi2_median": chi2_median,
            "npz_path": str(npz_path),
            "fits_path": str(fits_path),
            "message": maps.message,
        }
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
        append_run_log(log_path, f"FAIL cube={cube_path} message={message}")
        return {
            "cube_path": str(cube_path),
            "status": "failed",
            "n_spaxels_fitted": 0,
            "n_quality_ok": 0,
            "snr_median": "",
            "chi2_median": "",
            "npz_path": str(npz_path),
            "fits_path": str(fits_path),
            "message": message,
        }


def collect_cube_paths(
    cube: str | Path | None = None,
    cube_glob: str | None = None,
    manifest: str | Path | None = None,
    limit: int | None = None,
) -> list[Path]:
    paths: list[Path] = []
    if cube is not None:
        paths.append(Path(cube))
    if cube_glob:
        paths.extend(Path(path) for path in sorted(glob(cube_glob)))
    if manifest is not None:
        paths.extend(read_manifest_cube_paths(manifest))
    if not paths:
        raise ValueError("No cube paths found. Check --cube, --cube-glob, or --manifest")
    unique = unique_paths(paths)
    if limit is not None:
        if int(limit) < 1:
            raise ValueError("--limit must be >= 1")
        unique = unique[: int(limit)]
    return unique


def _progress_message(completed: int, total: int, rows: list[dict[str, object]]) -> str:
    n_ok = sum(row.get("status") == "ok" for row in rows)
    n_failed = sum(row.get("status") == "failed" for row in rows)
    n_skipped = sum(row.get("status") == "skipped" for row in rows)
    return (
        f"[kinematic_moments] processed {completed}/{total} galaxies "
        f"(ok={n_ok}, failed={n_failed}, skipped={n_skipped})"
    )


def _cube_status_message(completed: int, total: int, row: dict[str, object]) -> str:
    cube_path = Path(str(row.get("cube_path", "")))
    name = cube_path.name or str(row.get("cube_path", ""))
    status = row.get("status", "")
    fitted = row.get("n_spaxels_fitted", "")
    quality = row.get("n_quality_ok", "")
    message_lines = str(row.get("message", "") or "").splitlines()
    message = message_lines[0] if message_lines else ""
    detail = (
        f"[kinematic_moments] cube {completed}/{total} status={status} "
        f"name={name} fitted={fitted} quality={quality}"
    )
    if message:
        detail += f" message={message}"
    return detail


def _maybe_report_progress(
    completed: int,
    total: int,
    rows: list[dict[str, object]],
    progress_every: int,
    log_path: str | Path | None = None,
) -> None:
    if progress_every <= 0:
        return
    if completed % progress_every == 0 or completed == total:
        message = _progress_message(completed, total, rows)
        print(message, flush=True)
        append_run_log(log_path, message)


def _report_cube_status(
    completed: int,
    total: int,
    row: dict[str, object],
    log_path: str | Path | None = None,
) -> None:
    message = _cube_status_message(completed, total, row)
    print(message, flush=True)
    append_run_log(log_path, message)


def process_catalog(
    cube_paths: Iterable[Path],
    output_dir: str | Path,
    config: KinematicMomentsConfig,
    n_workers: int = 1,
    max_spaxels: int | None = None,
    overwrite: bool = False,
    progress_every: int = 10,
    log_path: str | Path | None = None,
) -> list[dict[str, object]]:
    cube_paths = list(cube_paths)
    append_run_log(
        log_path,
        f"BATCH START n_cubes={len(cube_paths)} n_workers={int(n_workers)} "
        f"max_spaxels={max_spaxels} overwrite={overwrite}",
    )
    if int(n_workers) <= 1:
        rows: list[dict[str, object]] = []
        total = len(cube_paths)
        for completed, path in enumerate(cube_paths, start=1):
            row = process_cube(
                path,
                output_dir,
                config,
                max_spaxels=max_spaxels,
                overwrite=overwrite,
                show_progress=(total == 1),
                log_path=log_path,
            )
            rows.append(row)
            _report_cube_status(completed, total, row, log_path)
            _maybe_report_progress(completed, total, rows, progress_every, log_path)
        append_run_log(log_path, "BATCH END " + _progress_message(total, total, rows))
        return rows

    rows = []
    with ProcessPoolExecutor(max_workers=int(n_workers)) as executor:
        futures = {
            executor.submit(
                process_cube,
                path,
                output_dir,
                config,
                max_spaxels,
                overwrite,
                False,
                log_path,
            ): path
            for path in cube_paths
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            cube_path = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                npz_path, fits_path = output_paths(cube_path, output_dir)
                message = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
                append_run_log(log_path, f"WORKER FAIL cube={cube_path} message={message}")
                row = {
                    "cube_path": str(Path(cube_path).expanduser().resolve()),
                    "status": "failed",
                    "n_spaxels_fitted": 0,
                    "n_quality_ok": 0,
                    "snr_median": "",
                    "chi2_median": "",
                    "npz_path": str(npz_path),
                    "fits_path": str(fits_path),
                    "message": message,
                }
            rows.append(row)
            _report_cube_status(completed, len(cube_paths), row, log_path)
            _maybe_report_progress(completed, len(cube_paths), rows, progress_every, log_path)
    append_run_log(log_path, "BATCH END " + _progress_message(len(cube_paths), len(cube_paths), rows))
    return rows


def write_manifest(
    rows: list[dict[str, object]],
    output_dir: str | Path,
    log_path: str | Path | None = None,
) -> Path:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "kinematics_manifest.csv"
    fieldnames = [
        "cube_path",
        "status",
        "n_spaxels_fitted",
        "n_quality_ok",
        "snr_median",
        "chi2_median",
        "npz_path",
        "fits_path",
        "message",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    append_run_log(log_path, f"MANIFEST written path={path} rows={len(rows)}")
    return path
