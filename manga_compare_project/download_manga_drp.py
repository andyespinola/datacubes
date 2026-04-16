import argparse
from pathlib import Path
from urllib.request import urlopen


SAS_BASE = "https://data.sdss.org/sas"


def build_url(release, drpver, plateifu, product):
    if "-" not in plateifu:
        raise ValueError("plateifu debe tener formato PLATE-IFU, por ejemplo 7443-12703")
    plate, _ifu = plateifu.split("-", 1)
    return (
        f"{SAS_BASE}/{release}/manga/spectro/redux/"
        f"{drpver}/{plate}/stack/manga-{plateifu}-{product}.fits.gz"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Descarga un producto DRP de MaNGA desde el SAS oficial de SDSS"
    )
    parser.add_argument("--plateifu", default="7443-12703", help="Identificador plate-ifu")
    parser.add_argument(
        "--product",
        default="LOGCUBE",
        choices=["LOGCUBE", "LINCUBE", "LOGRSS", "LINRSS"],
        help="Producto MaNGA DRP a descargar",
    )
    parser.add_argument("--release", default="dr17", help="Release SDSS, por ejemplo dr17")
    parser.add_argument(
        "--drpver",
        default="v3_1_1",
        help="Version DRP usada en la ruta SAS, por ejemplo v3_1_1",
    )
    parser.add_argument(
        "--outdir",
        default="data",
        help="Directorio donde guardar el FITS descargado",
    )
    parser.add_argument("--force", action="store_true", help="Sobrescribe si ya existe")
    parser.add_argument("--dry-run", action="store_true", help="Solo imprime la URL")
    return parser.parse_args()


def download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def main():
    args = parse_args()
    url = build_url(args.release, args.drpver, args.plateifu, args.product)
    outdir = Path(args.outdir).resolve()
    destination = outdir / f"manga-{args.plateifu}-{args.product}.fits.gz"

    print(f"URL: {url}")
    print(f"Destino: {destination}")

    if args.dry_run:
        return

    if destination.exists() and not args.force:
        print("El archivo ya existe. Usa --force si quieres re-descargarlo.")
        return

    download_file(url, destination)
    print("Descarga completada.")


if __name__ == "__main__":
    main()
