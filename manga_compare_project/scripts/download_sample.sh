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
MANGA_RELEASE="${MANGA_RELEASE:-dr17}"
MANGA_DRP_VER="${MANGA_DRP_VER:-v3_1_1}"
MANGA_PLATEIFU="${MANGA_PLATEIFU:-7443-12703}"
MANGA_PRODUCT="${MANGA_PRODUCT:-LOGCUBE}"
DATA_DIR="${DATA_DIR:-./data}"

"$VENV_DIR/bin/python" download_manga_drp.py \
  --release "$MANGA_RELEASE" \
  --drpver "$MANGA_DRP_VER" \
  --plateifu "$MANGA_PLATEIFU" \
  --product "$MANGA_PRODUCT" \
  --outdir "$DATA_DIR" \
  "$@"
