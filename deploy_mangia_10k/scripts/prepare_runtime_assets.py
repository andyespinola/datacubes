#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from urllib.request import urlopen


ROOT_DIR = Path(__file__).resolve().parent.parent
CATALOG_TARGET = ROOT_DIR / "MaNGIA_catalog.fits"
TEMPLATE_TARGET = ROOT_DIR / "official_mangia" / "libs" / "MaStar_CB19.slog_1_5.fits.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materializa los assets externos minimos para correr deploy_mangia_10k"
    )
    parser.add_argument("--catalog-source", default=os.environ.get("CATALOG_SOURCE_PATH", ""))
    parser.add_argument("--catalog-url", default=os.environ.get("CATALOG_URL", ""))
    parser.add_argument("--template-source", default=os.environ.get("TEMPLATE_SOURCE_PATH", ""))
    parser.add_argument("--template-url", default=os.environ.get("TEMPLATE_URL", ""))
    parser.add_argument(
        "--workspace-root",
        default=os.environ.get("WORKSPACE_ROOT", str(ROOT_DIR.parent)),
        help="Workspace padre opcional para intentar copiar assets ya presentes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-copia o re-descarga aunque el archivo target ya exista",
    )
    return parser.parse_args()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _normalize_path(raw: str) -> Path | None:
    raw = raw.strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _candidate_paths(explicit_source: str, workspace_root: Path, filename: str) -> list[Path]:
    candidates: list[Path] = []
    explicit = _normalize_path(explicit_source)
    if explicit is not None:
        candidates.append(explicit)

    candidates.extend(
        [
            ROOT_DIR / filename,
            workspace_root / filename,
            workspace_root / "official_mangia" / "libs" / filename,
            workspace_root / "deploy_mangia_10k" / filename,
            workspace_root / "deploy_mangia_10k" / "official_mangia" / "libs" / filename,
        ]
    )

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


def ensure_asset(
    *,
    label: str,
    target: Path,
    explicit_source: str,
    url: str,
    workspace_root: Path,
    filename: str,
    force: bool,
) -> dict[str, str]:
    if target.exists() and not force:
        return {"status": "existing", "path": str(target.resolve())}

    if force and target.exists():
        target.unlink()

    for candidate in _candidate_paths(explicit_source, workspace_root, filename):
        if candidate.exists() and candidate.is_file() and candidate != target.resolve():
            _copy(candidate, target)
            return {
                "status": "copied",
                "path": str(target.resolve()),
                "source": str(candidate),
            }

    if url.strip():
        _download(url.strip(), target)
        return {
            "status": "downloaded",
            "path": str(target.resolve()),
            "source": url.strip(),
        }

    raise FileNotFoundError(
        f"No pude materializar {label}. "
        f"Busque una copia local de {filename} y no la encontre. "
        f"Define una ruta local o URL via variables de entorno y reintenta."
    )


def main() -> None:
    args = parse_args()
    workspace_root = Path(args.workspace_root).expanduser().resolve()

    result = {
        "catalog": ensure_asset(
            label="MaNGIA_catalog.fits",
            target=CATALOG_TARGET,
            explicit_source=args.catalog_source,
            url=args.catalog_url,
            workspace_root=workspace_root,
            filename="MaNGIA_catalog.fits",
            force=args.force,
        ),
        "template_ssp": ensure_asset(
            label="MaStar_CB19.slog_1_5.fits.gz",
            target=TEMPLATE_TARGET,
            explicit_source=args.template_source,
            url=args.template_url,
            workspace_root=workspace_root,
            filename="MaStar_CB19.slog_1_5.fits.gz",
            force=args.force,
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
