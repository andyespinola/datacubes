#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
LOG_DIR="${LOG_DIR:-./logs}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
RSS_INPUT_DIR="${RSS_INPUT_DIR:-./rss_input}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -r requirements.txt
"$VENV_DIR/bin/python" scripts/prepare_runtime_assets.py

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$RSS_INPUT_DIR"

echo "Entorno listo en: $ROOT_DIR/$VENV_DIR"
echo "RSS dir: $RSS_INPUT_DIR"
echo "Output dir: $OUTPUT_DIR"
