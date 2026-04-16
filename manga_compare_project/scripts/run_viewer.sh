#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

VENV_DIR="${VENV_DIR:-.venv}"
VIEWER_HOST="${VIEWER_HOST:-0.0.0.0}"
VIEWER_PORT="${VIEWER_PORT:-8010}"
VIEWER_DATA_DIR="${VIEWER_DATA_DIR:-./data}"

"$VENV_DIR/bin/python" manga_compare_viewer.py \
  --host "$VIEWER_HOST" \
  --port "$VIEWER_PORT" \
  --data-dir "$VIEWER_DATA_DIR" \
  "$@"
