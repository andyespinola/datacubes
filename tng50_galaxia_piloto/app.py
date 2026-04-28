from __future__ import annotations

import argparse

from pilot_viewer.web import create_app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Viewer web para la galaxia piloto TNG50")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5051)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
