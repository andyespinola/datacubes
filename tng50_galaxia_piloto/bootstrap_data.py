from __future__ import annotations

import argparse
import json

from pilot_viewer.downloader import bootstrap_pilot_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Materializa los archivos del caso piloto TNG50 para el viewer")
    parser.add_argument("--force-download", action="store_true", help="Ignora el cache local e intenta bajar desde TNG")
    parser.add_argument("--api-key", default="", help="API key de TNG opcional")
    args = parser.parse_args()

    result = bootstrap_pilot_data(force_download=args.force_download, api_key=args.api_key)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
