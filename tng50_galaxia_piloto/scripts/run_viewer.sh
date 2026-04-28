#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_PYTHON="$ROOT_DIR/../.venv/bin/python"
if [ -x "$DEFAULT_PYTHON" ]; then
  PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5051}"

cd "$ROOT_DIR"
"$PYTHON_BIN" app.py --host "$HOST" --port "$PORT"
