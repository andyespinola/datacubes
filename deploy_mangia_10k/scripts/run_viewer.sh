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
VIEWER_PORT="${VIEWER_PORT:-8000}"

"$VENV_DIR/bin/python" cube_web_viewer.py --host "$VIEWER_HOST" --port "$VIEWER_PORT" "$@"
