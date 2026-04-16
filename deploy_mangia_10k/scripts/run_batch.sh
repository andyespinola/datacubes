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
RSS_INPUT_DIR="${RSS_INPUT_DIR:-./rss_input}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
CATALOG_PATH="${CATALOG_PATH:-./MaNGIA_catalog.fits}"
TEMPLATE_SSP_CONTROL="${TEMPLATE_SSP_CONTROL:-./official_mangia/libs/MaStar_CB19.slog_1_5.fits.gz}"
RSS_GLOB="${RSS_GLOB:-*.cube_RSS.fits*}"
BATCH_WORKERS="${BATCH_WORKERS:-1}"
START_INDEX="${START_INDEX:-0}"
COUNT="${COUNT:-0}"
NOISE_SN="${NOISE_SN:-5.0}"
NOISE_RADIUS="${NOISE_RADIUS:-2.0}"
THET="${THET:-0.0}"
INCLUDE_GAS="${INCLUDE_GAS:-0}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
LOG_DIR="${LOG_DIR:-./logs}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

ARGS=(
  --rss-dir "$RSS_INPUT_DIR"
  --output-dir "$OUTPUT_DIR"
  --catalog "$CATALOG_PATH"
  --template-ssp-control "$TEMPLATE_SSP_CONTROL"
  --rss-glob "$RSS_GLOB"
  --workers "$BATCH_WORKERS"
  --start-index "$START_INDEX"
  --count "$COUNT"
  --noise-sn "$NOISE_SN"
  --noise-radius "$NOISE_RADIUS"
  --thet "$THET"
)

if [[ "$INCLUDE_GAS" == "1" ]]; then
  ARGS+=(--include-gas)
fi

if [[ "$FORCE_REBUILD" == "1" ]]; then
  ARGS+=(--force)
fi

"$VENV_DIR/bin/python" batch_reconstruct.py "${ARGS[@]}" "$@"
