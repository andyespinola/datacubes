from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "preview_viewer_static"
DEFAULT_PREVIEW_DIR = BASE_DIR / "data" / "output" / "previews"
PREVIEW_DIR_CANDIDATES = (
    BASE_DIR / "previews",
    DEFAULT_PREVIEW_DIR,
    BASE_DIR / "data" / "output" / "mangia" / "previews",
    BASE_DIR / "data" / "output" / "manga" / "previews",
    BASE_DIR / "data" / "output" / "portable_catalog" / "previews_from_script",
)
DEFAULT_PAGE_SIZE = 96
MAX_PAGE_SIZE = 240
SORT_KEYS = {"name", "mtime_desc", "mtime_asc", "size_desc", "size_asc"}


@dataclass(frozen=True)
class PreviewItem:
    path: str
    name: str
    stem: str
    size: int
    modified: float


def contains_png(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    try:
        return any(child.is_file() and child.suffix.lower() == ".png" for child in path.rglob("*"))
    except OSError:
        return False


def default_preview_dir() -> Path:
    env_value = os.environ.get("MANGIA_PREVIEW_DIR")
    if env_value:
        return Path(env_value).expanduser()

    for candidate in PREVIEW_DIR_CANDIDATES:
        if contains_png(candidate):
            return candidate

    for candidate in PREVIEW_DIR_CANDIDATES:
        if candidate.exists():
            return candidate

    return DEFAULT_PREVIEW_DIR


def scan_preview_items(root: Path) -> list[PreviewItem]:
    root = root.resolve()
    if not root.exists() or not root.is_dir():
        return []

    items: list[PreviewItem] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        try:
            stat = path.stat()
            relative_path = path.relative_to(root).as_posix()
        except OSError:
            continue
        items.append(
            PreviewItem(
                path=relative_path,
                name=path.name,
                stem=path.stem,
                size=int(stat.st_size),
                modified=float(stat.st_mtime),
            )
        )
    return sorted(items, key=lambda item: item.path.casefold())


class PreviewCatalog:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self._items: list[PreviewItem] = []
        self._scanned_at = 0.0
        self._lock = threading.Lock()

    @property
    def exists(self) -> bool:
        return self.root.exists() and self.root.is_dir()

    @property
    def scanned_at(self) -> float:
        return self._scanned_at

    def refresh(self) -> list[PreviewItem]:
        items = scan_preview_items(self.root)
        with self._lock:
            self._items = items
            self._scanned_at = time.time()
        return items

    def items(self) -> list[PreviewItem]:
        with self._lock:
            if self._scanned_at > 0:
                return list(self._items)
        return self.refresh()


def clamp_int(value: str | None, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def filter_items(items: list[PreviewItem], query: str) -> list[PreviewItem]:
    terms = [term.casefold() for term in query.split() if term.strip()]
    if not terms:
        return items
    return [
        item
        for item in items
        if all(term in item.path.casefold() or term in item.stem.casefold() for term in terms)
    ]


def sort_items(items: list[PreviewItem], sort_key: str) -> list[PreviewItem]:
    if sort_key == "mtime_desc":
        return sorted(items, key=lambda item: (item.modified, item.path.casefold()), reverse=True)
    if sort_key == "mtime_asc":
        return sorted(items, key=lambda item: (item.modified, item.path.casefold()))
    if sort_key == "size_desc":
        return sorted(items, key=lambda item: (item.size, item.path.casefold()), reverse=True)
    if sort_key == "size_asc":
        return sorted(items, key=lambda item: (item.size, item.path.casefold()))
    return sorted(items, key=lambda item: item.path.casefold())


def relative_png_path(catalog: PreviewCatalog, relative_path: str) -> str:
    if not relative_path:
        abort(400, "Falta la ruta de la imagen")
    if Path(relative_path).suffix.lower() != ".png":
        abort(404, "Solo se sirven previews PNG")

    resolved = (catalog.root / relative_path).resolve()
    try:
        resolved.relative_to(catalog.root)
    except ValueError:
        abort(404, "La ruta solicitada queda fuera del directorio de previews")
    if not resolved.exists() or not resolved.is_file():
        abort(404, "No existe la imagen solicitada")
    return resolved.relative_to(catalog.root).as_posix()


def create_app(preview_dir: str | Path | None = None) -> Flask:
    catalog = PreviewCatalog(Path(preview_dir) if preview_dir is not None else default_preview_dir())
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/api/config")
    def api_config():
        items = catalog.items()
        return jsonify(
            {
                "preview_dir": catalog.root.as_posix(),
                "exists": catalog.exists,
                "total": len(items),
                "scanned_at": catalog.scanned_at,
                "default_page_size": DEFAULT_PAGE_SIZE,
                "max_page_size": MAX_PAGE_SIZE,
            }
        )

    @app.get("/api/images")
    def api_images():
        if request.args.get("refresh") == "1":
            catalog.refresh()

        offset = clamp_int(request.args.get("offset"), 0, 0, 10**9)
        limit = clamp_int(request.args.get("limit"), DEFAULT_PAGE_SIZE, 1, MAX_PAGE_SIZE)
        query = request.args.get("q", "").strip()
        sort_key = request.args.get("sort", "name")
        if sort_key not in SORT_KEYS:
            sort_key = "name"

        all_items = catalog.items()
        filtered = sort_items(filter_items(all_items, query), sort_key)
        total_filtered = len(filtered)
        if total_filtered == 0:
            offset = 0
        elif offset >= total_filtered:
            offset = max(0, ((total_filtered - 1) // limit) * limit)

        page_items = filtered[offset : offset + limit]
        return jsonify(
            {
                "preview_dir": catalog.root.as_posix(),
                "exists": catalog.exists,
                "total": len(all_items),
                "filtered_total": total_filtered,
                "offset": offset,
                "limit": limit,
                "query": query,
                "sort": sort_key,
                "scanned_at": catalog.scanned_at,
                "items": [asdict(item) for item in page_items],
            }
        )

    @app.post("/api/refresh")
    def api_refresh():
        items = catalog.refresh()
        return jsonify(
            {
                "preview_dir": catalog.root.as_posix(),
                "exists": catalog.exists,
                "total": len(items),
                "scanned_at": catalog.scanned_at,
            }
        )

    @app.get("/previews/<path:relative_path>")
    def preview_file(relative_path: str):
        safe_path = relative_png_path(catalog, relative_path)
        return send_from_directory(
            catalog.root,
            safe_path,
            conditional=True,
            max_age=3600,
        )

    return app


app = create_app()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualizador web de previews PNG de ImagesMangGenerator.")
    parser.add_argument(
        "--preview-dir",
        default=None,
        help=(
            "Directorio con previews PNG. Por defecto usa MANGIA_PREVIEW_DIR o detecta "
            "una carpeta previews conocida dentro de ImagesMangGenerator."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5052)
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    viewer = create_app(args.preview_dir)
    viewer.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
