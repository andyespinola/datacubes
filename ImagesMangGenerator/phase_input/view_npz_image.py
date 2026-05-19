from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


def load_npz_image(path: str | Path) -> tuple[np.ndarray, dict]:
    with np.load(path) as payload:
        image = np.asarray(payload["image"], dtype=np.float32)
        metadata = json.loads(payload["metadata"].item()) if "metadata" in payload else {}
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Esperaba image con shape (3,H,W), encontre {image.shape}")
    return image, metadata


def make_rgb_preview(
    image: np.ndarray,
    stretch: str = "asinh",
    percentile: float = 99.5,
    smooth_sigma: float = 0.85,
    preview_size: int = 768,
    rgb_order: str = "irg",
) -> np.ndarray:
    rgb_order = rgb_order.lower()
    band_index = {"g": 0, "r": 1, "i": 2}
    if sorted(rgb_order) != ["g", "i", "r"]:
        raise ValueError("--rgb-order debe contener exactamente las bandas g, r, i")

    rgb = np.stack([image[band_index[band]] for band in rgb_order], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    if smooth_sigma > 0:
        rgb = gaussian_filter(rgb, sigma=(smooth_sigma, smooth_sigma, 0.0), mode="nearest")

    low = float(np.nanpercentile(rgb, 1.0))
    high = float(np.nanpercentile(rgb, percentile))
    if high <= low:
        high = float(np.nanmax(rgb))
    if high <= low:
        return np.zeros_like(rgb, dtype=np.float32)

    scaled = np.clip((rgb - low) / (high - low), 0.0, None)
    if stretch == "linear":
        preview = scaled
    elif stretch == "sqrt":
        preview = np.sqrt(scaled)
    elif stretch == "asinh":
        preview = np.arcsinh(10.0 * scaled) / np.arcsinh(10.0)
    else:
        raise ValueError(f"stretch no soportado: {stretch}")
    preview = np.clip(preview, 0.0, 1.0).astype(np.float32)

    if preview_size > 0:
        height, width = preview.shape[:2]
        scale = float(preview_size) / max(height, width)
        if scale > 1.0:
            preview = zoom(preview, (scale, scale, 1.0), order=3)
            preview = np.clip(preview, 0.0, 1.0).astype(np.float32)
    return preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualiza o exporta NPZ de ImagesMangGenerator.")
    parser.add_argument("npz_path", help="Archivo NPZ generado por ImageProvider.")
    parser.add_argument("--out", help="Ruta PNG de salida. Si se omite, abre una ventana matplotlib.")
    parser.add_argument("--stretch", default="asinh", choices=["asinh", "sqrt", "linear"])
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument("--smooth-sigma", type=float, default=0.85, help="Suavizado gaussiano en spaxels/pixeles.")
    parser.add_argument("--preview-size", type=int, default=768, help="Tamaño del lado mayor del PNG exportado.")
    parser.add_argument("--rgb-order", default="irg", help="Orden de bandas para RGB. Default: i,r,g.")
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    image, metadata = load_npz_image(args.npz_path)
    rgb = make_rgb_preview(
        image,
        stretch=args.stretch,
        percentile=args.percentile,
        smooth_sigma=args.smooth_sigma,
        preview_size=args.preview_size,
        rgb_order=args.rgb_order,
    )
    title = metadata.get("galaxy_id", Path(args.npz_path).stem)
    source = metadata.get("source")
    unit = metadata.get("unit")
    if source or unit:
        title = f"{title} | {source or '?'} | {unit or '?'}"

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(out, rgb, origin="lower")
        print(out)
    else:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=args.dpi)
        ax.imshow(rgb, origin="lower", interpolation="bicubic")
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
        fig.tight_layout(pad=0.1)
        plt.show()


if __name__ == "__main__":
    main()
