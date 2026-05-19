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
fi

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

PREVIEW_INPUT_DIR="${PREVIEW_INPUT_DIR:-${MANGIA_OUTPUT_DIR:-}}"
if [[ -z "${PREVIEW_INPUT_DIR}" ]]; then
  echo "Configura PREVIEW_INPUT_DIR o MANGIA_OUTPUT_DIR en ${ENV_FILE}."
  exit 1
fi

PREVIEW_OUTPUT_DIR="${PREVIEW_OUTPUT_DIR:-${PREVIEW_INPUT_DIR}/previews}"
PREVIEW_PATTERN="${PREVIEW_PATTERN:-*.npz}"
PREVIEW_RECURSIVE="${PREVIEW_RECURSIVE:-0}"
PREVIEW_OVERWRITE="${PREVIEW_OVERWRITE:-0}"
PREVIEW_STRETCH="${PREVIEW_STRETCH:-asinh}"
PREVIEW_PERCENTILE="${PREVIEW_PERCENTILE:-99.5}"
PREVIEW_SMOOTH_SIGMA="${PREVIEW_SMOOTH_SIGMA:-0.85}"
PREVIEW_SIZE="${PREVIEW_SIZE:-768}"
PREVIEW_RGB_ORDER="${PREVIEW_RGB_ORDER:-irg}"

mkdir -p "${PREVIEW_OUTPUT_DIR}"
cd "${PROJECT_DIR}"

echo "Input NPZ: ${PREVIEW_INPUT_DIR}"
echo "Output PNG: ${PREVIEW_OUTPUT_DIR}"
echo "Python: ${PYTHON_BIN}"

count=0
skipped=0
failed=0

if [[ "${PREVIEW_RECURSIVE}" == "1" || "${PREVIEW_RECURSIVE}" == "true" ]]; then
  find_args=("${PREVIEW_INPUT_DIR}" -type f -name "${PREVIEW_PATTERN}")
else
  find_args=("${PREVIEW_INPUT_DIR}" -maxdepth 1 -type f -name "${PREVIEW_PATTERN}")
fi

while IFS= read -r -d '' npz_path; do
  base="$(basename "${npz_path}" .npz)"
  png_path="${PREVIEW_OUTPUT_DIR}/${base}.png"

  if [[ -f "${png_path}" && "${PREVIEW_OVERWRITE}" != "1" && "${PREVIEW_OVERWRITE}" != "true" ]]; then
    skipped=$((skipped + 1))
    continue
  fi

  if "${PYTHON_BIN}" -m ImagesMangGenerator.phase_input.view_npz_image \
    "${npz_path}" \
    --out "${png_path}" \
    --stretch "${PREVIEW_STRETCH}" \
    --percentile "${PREVIEW_PERCENTILE}" \
    --smooth-sigma "${PREVIEW_SMOOTH_SIGMA}" \
    --preview-size "${PREVIEW_SIZE}" \
    --rgb-order "${PREVIEW_RGB_ORDER}" >/dev/null; then
    count=$((count + 1))
  else
    failed=$((failed + 1))
    echo "Fallo preview: ${npz_path}" >&2
  fi
done < <(find "${find_args[@]}" -print0 | sort -z)

echo "Previews generados: ${count}"
echo "Previews saltados: ${skipped}"
echo "Previews fallidos: ${failed}"

if [[ "${failed}" -gt 0 ]]; then
  exit 1
fi

