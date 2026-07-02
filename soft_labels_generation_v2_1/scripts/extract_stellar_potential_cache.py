from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
WORKSPACE_DIR = PROJECT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aperturenet_labels.config import PipelineConfig, resolve_project_path  # noqa: E402
from aperturenet_labels.io.assets import discover_local_assets  # noqa: E402
from aperturenet_labels.io.tng_potential import (  # noqa: E402
    check_url,
    extract_stellar_potential_cache,
    offsets_url,
    snapshot_chunk_url,
)


def load_env_file(path: str | Path | None) -> None:
    if not path:
        return
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _resolve_workspace_path(value: str, default: Path) -> Path:
    if not value:
        return default
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_DIR / path).resolve()


def _selected_galaxy_ids(args: argparse.Namespace, config: PipelineConfig) -> list[str]:
    if args.all_local:
        return [asset.galaxy_id for asset in discover_local_assets(config.data)]
    if args.galaxy_id:
        return list(args.galaxy_id)
    raise SystemExit("Usa --galaxy-id o --all-local")


def _interpret_check(check: dict) -> str:
    status = int(check.get("status_code", 0))
    url = str(check.get("url", ""))
    if status == 200:
        return "ok"
    if status == 403 and "snapshot-" in url:
        return "forbidden_full_snapshot_file; verify TNG account permissions for full snapshot downloads"
    if status in {503, 504} and "offsets." in url:
        return "offsets_endpoint_unavailable_or_timed_out; retry later or download offsets by another route"
    if status >= 500:
        return "server_error; retry later"
    if status >= 400:
        return "client_or_permission_error"
    if status == 0:
        return "request_error"
    return "unexpected_status"


def _preflight(galaxy_ids: list[str], simulation: str, api_key: str) -> dict:
    snapshots = sorted({int(galaxy_id.split("-")[1]) for galaxy_id in galaxy_ids})
    checks = []
    for snapshot in snapshots:
        checks.append(check_url(offsets_url(simulation, snapshot), api_key))
        checks.append(check_url(snapshot_chunk_url(simulation, snapshot, 0), api_key))
    for check in checks:
        check["interpretation"] = _interpret_check(check)
    return {"simulation": simulation, "snapshots": snapshots, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description="Extrae cache local de PartType4/Potential desde chunks completos de TNG.")
    parser.add_argument("--config", default="", help="YAML config path relative to soft_labels_generation_v2_1/")
    parser.add_argument("--galaxy-id", action="append", default=[], help="Galaxy id, e.g. TNG50-87-155298; can repeat")
    parser.add_argument("--all-local", action="store_true", help="Procesa las galaxias locales configuradas")
    parser.add_argument("--data-dir", default="", help="Directorio data/ con cutout_phase2")
    parser.add_argument("--snapshot-cache-dir", default="../data/tng_snapshot_cache", help="Cache para offsets y chunks snapshot")
    parser.add_argument("--outdir", default="../data/potential_cache", help="Directorio de salida para potencial por galaxia")
    parser.add_argument("--env-file", default="../orientation_projection_validation/.env", help="Archivo .env con TNG_API_KEY")
    parser.add_argument("--api-key", default="", help="TNG API key; si se omite usa TNG_API_KEY")
    parser.add_argument("--simulation", default="TNG50-1", help="Nombre de simulación TNG")
    parser.add_argument("--force-download", action="store_true", help="Vuelve a descargar offsets/chunks aunque existan")
    parser.add_argument("--overwrite", action="store_true", help="Reescribe caches de potencial existentes")
    parser.add_argument("--download-retries", type=int, default=3, help="Reintentos por archivo grande")
    parser.add_argument("--download-backoff-seconds", type=float, default=10.0, help="Backoff base entre reintentos")
    parser.add_argument("--preflight", action="store_true", help="Solo verifica disponibilidad de endpoints, sin descargar archivos grandes")
    args = parser.parse_args()

    load_env_file(_resolve_workspace_path(args.env_file, WORKSPACE_DIR / "orientation_projection_validation" / ".env"))
    api_key = args.api_key or os.environ.get("TNG_API_KEY", "")
    if not api_key:
        raise SystemExit("Necesito TNG_API_KEY en entorno, --api-key o --env-file")

    config = PipelineConfig.from_yaml(args.config)
    if args.data_dir:
        config.data.data_dir = resolve_project_path(args.data_dir)
    galaxy_ids = _selected_galaxy_ids(args, config)
    if args.preflight:
        print(json.dumps(_preflight(galaxy_ids, args.simulation, api_key), indent=2, sort_keys=True))
        return 0

    snapshot_cache_dir = _resolve_workspace_path(args.snapshot_cache_dir, WORKSPACE_DIR / "data" / "tng_snapshot_cache")
    outdir = _resolve_workspace_path(args.outdir, WORKSPACE_DIR / "data" / "potential_cache")
    rows = []
    for galaxy_id in galaxy_ids:
        result = extract_stellar_potential_cache(
            galaxy_id=galaxy_id,
            data_dir=config.data.data_dir,
            snapshot_cache_dir=snapshot_cache_dir,
            output_dir=outdir,
            api_key=api_key,
            simulation=args.simulation,
            force_download=args.force_download,
            overwrite=args.overwrite,
            download_retries=args.download_retries,
            download_backoff_seconds=args.download_backoff_seconds,
        )
        rows.append(
            {
                "galaxy_id": result.galaxy_id,
                "snapshot": result.snapshot,
                "subhalo_id": result.subhalo_id,
                "output_path": str(result.output_path),
                "n_particles": result.n_particles,
                "chunks_used": list(result.chunks_used),
                "offset_path": str(result.offset_path),
                "elapsed_seconds": result.elapsed_seconds,
            }
        )
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
