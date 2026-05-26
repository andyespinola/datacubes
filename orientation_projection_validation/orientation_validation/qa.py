from __future__ import annotations

from pathlib import Path
import struct
import zlib

import numpy as np

from .paths import ensure_structural_labeling_on_path

ensure_structural_labeling_on_path()

from labeling.constants import CLASS_INDEX, PHYSICAL_CLASS_INDICES  # noqa: E402


CLASS_RGB = {
    CLASS_INDEX["no_valido"]: (18, 18, 18),
    CLASS_INDEX["bulbo"]: (226, 84, 74),
    CLASS_INDEX["disco"]: (75, 146, 219),
    CLASS_INDEX["barra"]: (238, 178, 67),
    CLASS_INDEX["brazos"]: (78, 178, 122),
    CLASS_INDEX["other"]: (152, 111, 196),
    CLASS_INDEX["incierto"]: (155, 155, 155),
}


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def write_png(path: str | Path, rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(np.clip(rgb, 0, 255), dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("write_png espera un array RGB uint8 [H,W,3]")
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    payload = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(raw, level=9)),
            _png_chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(payload)


def hard_rgb_from_soft(soft: np.ndarray, mval: np.ndarray) -> np.ndarray:
    hard_physical = np.argmax(soft[list(PHYSICAL_CLASS_INDICES)], axis=0)
    physical_indices = np.asarray(list(PHYSICAL_CLASS_INDICES), dtype=np.int16)
    hard = physical_indices[hard_physical]
    hard[mval == 0] = CLASS_INDEX["no_valido"]
    hard[mval == 2] = CLASS_INDEX["incierto"]
    rgb = np.zeros((*hard.shape, 3), dtype=np.uint8)
    for idx, color in CLASS_RGB.items():
        rgb[hard == idx] = color
    return rgb


def write_qa_mosaic(path: str | Path, products: dict[str, dict[str, np.ndarray]], variant: str = "Y_lum_psf") -> None:
    keys = sorted(products)
    if not keys:
        return
    tiles = [hard_rgb_from_soft(products[key][variant], products[key]["Mval"]) for key in keys]
    h, w, _ = tiles[0].shape
    pad = 3
    canvas = np.full((2 * h + pad, 2 * w + pad, 3), 255, dtype=np.uint8)
    positions = [(0, 0), (0, w + pad), (h + pad, 0), (h + pad, w + pad)]
    for tile, (y, x) in zip(tiles[:4], positions, strict=False):
        canvas[y : y + h, x : x + w] = tile
    write_png(path, canvas)

