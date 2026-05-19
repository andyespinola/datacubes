#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_DIR}"

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

if [[ ! -f ImagesMangGenerator/.env ]]; then
  cp ImagesMangGenerator/.env.example ImagesMangGenerator/.env
  echo "Creado ImagesMangGenerator/.env. Edita MANGIA_INPUT_DIR y MANGIA_OUTPUT_DIR antes de correr."
fi

echo "Entorno listo."

