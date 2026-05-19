#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${1:-${PROJECT_DIR}/ImagesMangGenerator/.env}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
else
  echo "No existe ${ENV_FILE}."
  echo "Copia ImagesMangGenerator/.env.example a ImagesMangGenerator/.env y ajusta las rutas."
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

MANGIA_INPUT_DIR="${MANGIA_INPUT_DIR:?Configura MANGIA_INPUT_DIR en ${ENV_FILE}}"
MANGIA_OUTPUT_DIR="${MANGIA_OUTPUT_DIR:?Configura MANGIA_OUTPUT_DIR en ${ENV_FILE}}"
MANGIA_PATTERN="${MANGIA_PATTERN:-*.cube.fits.gz}"
MANGIA_WORKERS="${MANGIA_WORKERS:-1}"
MANGIA_OUTPUT_SHAPE="${MANGIA_OUTPUT_SHAPE:-native}"
MANGIA_RECURSIVE="${MANGIA_RECURSIVE:-0}"
MANGIA_SKIP_EXISTING="${MANGIA_SKIP_EXISTING:-1}"

cmd=(
  "${PYTHON_BIN}" -m ImagesMangGenerator.phase_input.build_images catalog-mangia
  --input-dir "${MANGIA_INPUT_DIR}"
  --output-dir "${MANGIA_OUTPUT_DIR}"
  --pattern "${MANGIA_PATTERN}"
  --workers "${MANGIA_WORKERS}"
  --output-shape "${MANGIA_OUTPUT_SHAPE}"
)

if [[ "${MANGIA_RECURSIVE}" == "1" || "${MANGIA_RECURSIVE}" == "true" ]]; then
  cmd+=(--recursive)
fi

if [[ "${MANGIA_SKIP_EXISTING}" == "1" || "${MANGIA_SKIP_EXISTING}" == "true" ]]; then
  cmd+=(--skip-existing)
fi

if [[ -n "${MANGIA_LIMIT:-}" ]]; then
  cmd+=(--limit "${MANGIA_LIMIT}")
fi

if [[ -n "${MANGIA_MANIFEST:-}" ]]; then
  cmd+=(--manifest "${MANGIA_MANIFEST}")
fi

cd "${PROJECT_DIR}"
echo "Ejecutando:"
printf ' %q' "${cmd[@]}"
echo
exec "${cmd[@]}"

