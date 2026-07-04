#!/bin/bash
# Encadena, sin intervencion manual, todo lo que falta para dejar listas:
#   - las 20 galaxias de la oleada 1 (incluida TNG50-88-312423, la mas grande)
#   - la descarga completa (cutout+gas+metadata+fase2 DM) de las 80 de la oleada 2
# Pensado para correr desatendido durante horas. Cada paso es idempotente/resumible,
# asi que si algo se corta, volver a correr este script retoma donde quedo.
set -x
cd /home/andy/pythonProjects/datacubes/orientation_projection_validation

ENV_FILE=/home/andy/pythonProjects/datacubes/data/.env
CACHE=/media/andy/Data/tng
LOG_DIR=/media/andy/Data/tng/orchestrator_logs
mkdir -p "$LOG_DIR"

echo "=== $(date) INICIO orquestador ==="

# --- 1. Esperar a que termine el batch actual de la oleada 1 (proceso ya en curso) ---
echo "=== $(date) esperando a que termine el proceso wave1 en curso (PID 1686757) ==="
while kill -0 1686757 2>/dev/null; do sleep 20; done
echo "=== $(date) proceso wave1 anterior termino ==="

# --- 2. Restaurar TNG50-88-312423 y completar la oleada 1 con la version resumible/aislada ---
if [ -f /media/andy/Data/tng/mangia_flat_312423_setaside/TNG50-88-312423-0-127.cube.fits.gz ]; then
  mv /media/andy/Data/tng/mangia_flat_312423_setaside/TNG50-88-312423-0-127.cube.fits.gz \
     /media/andy/Data/tng/mangia_flat/
  echo "=== $(date) TNG50-88-312423 restaurada al directorio plano ==="
fi

cd /home/andy/pythonProjects/datacubes/soft_labels_generation_v2_2
python3 scripts/run_wave1.py --timeout-sec 3600 > "$LOG_DIR/wave1_final.log" 2>&1
echo "=== $(date) oleada 1 (20 galaxias) completa. Resumen: ==="
cat /media/andy/Data/tng/mangia_flat/wave1_run_summary.json

# --- 3. Esperar a que termine la descarga principal de la oleada 2 (proceso ya en curso) ---
cd /home/andy/pythonProjects/datacubes/orientation_projection_validation
echo "=== $(date) esperando a que termine la descarga principal wave2 en curso ==="
while pgrep -f "download_tng_assets.py.*wave2_manifest" > /dev/null; do sleep 20; done
echo "=== $(date) descarga principal wave2 (llamada original) termino ==="

# --- 4. Reintentar cutouts/metadata de wave2 que hayan quedado pendientes (varias pasadas) ---
for i in 1 2 3 4; do
  echo "=== $(date) pasada de reintento $i/4 para cutouts wave2 ==="
  python3 download_tng_assets.py \
    --manifest data/wave2_manifest.csv \
    --out-cache "$CACHE" \
    --env-file "$ENV_FILE" \
    --include-gas \
    --continue-on-error \
    --retry-failures 3 \
    --retry-delay-seconds 30 >> "$LOG_DIR/wave2_download_retry.log" 2>&1
done

# --- 5. Metadata de galaxias sin gas (el validador local las marca 'error' aunque el cutout este OK) ---
python3 - <<'PYEOF' >> "/media/andy/Data/tng/orchestrator_logs/wave2_gasless_fix.log" 2>&1
import csv, os, sys
sys.path.insert(0, '.')
from orientation_validation.download import download_json_atomic, metadata_url, validate_metadata
from orientation_validation.env import load_env_file

load_env_file('/home/andy/pythonProjects/datacubes/data/.env')
key = os.environ['TNG_API_KEY']

with open('data/wave2_manifest.csv', newline='') as f:
    rows = list(csv.DictReader(f))

fixed, failed = 0, 0
for r in rows:
    g = r['galaxy_id']
    snap, sub = int(r['snapshot']), int(r['subhalo_id'])
    cutout_path = f'/media/andy/Data/tng/cutouts/{g}.cutout.hdf5'
    meta_path = f'/media/andy/Data/tng/metadata/{g}.subhalo.json'
    if os.path.exists(cutout_path) and not os.path.exists(meta_path):
        try:
            download_json_atomic(metadata_url(snap, sub), meta_path, key, timeout=60)
            validate_metadata(meta_path)
            print('metadata OK (galaxia sin gas u otro corte):', g)
            fixed += 1
        except Exception as exc:
            print('metadata FALLO:', g, exc)
            failed += 1
print(f'arreglados={fixed} fallidos={failed}')
PYEOF
echo "=== $(date) fix de metadata para galaxias sin gas terminado ==="

# --- 6. Fase 2 (materia oscura) para las 80 de wave2 ---
for i in 1 2 3; do
  echo "=== $(date) pasada fase2 DM $i/3 para wave2 ==="
  python3 download_phase2_dm.py \
    --manifest data/wave2_manifest.csv \
    --out-cache "$CACHE" \
    --env-file "$ENV_FILE" \
    --continue-on-error >> "$LOG_DIR/wave2_phase2.log" 2>&1
done

# --- 7. Verificacion final de completitud ---
python3 - <<'PYEOF' > "/media/andy/Data/tng/orchestrator_logs/FINAL_STATUS.txt" 2>&1
import csv

def check(manifest_path):
    missing = []
    with open(manifest_path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        g = r['galaxy_id']
        for suffix, d in [
            ('.cutout.hdf5', 'cutouts'),
            ('.subhalo.json', 'metadata'),
            ('.cutout_phase2.hdf5', 'phase2'),
        ]:
            import os
            p = f'/media/andy/Data/tng/{d}/{g}{suffix}'
            if not os.path.exists(p):
                missing.append(p)
    return len(rows), missing

n1, miss1 = check('/home/andy/pythonProjects/datacubes/orientation_projection_validation/data/wave1_manifest.csv')
n2, miss2 = check('/home/andy/pythonProjects/datacubes/orientation_projection_validation/data/wave2_manifest.csv')
print(f'Oleada 1: {n1} galaxias, faltantes TNG: {len(miss1)}')
for m in miss1:
    print('  FALTA:', m)
print(f'Oleada 2: {n2} galaxias, faltantes TNG: {len(miss2)}')
for m in miss2:
    print('  FALTA:', m)

import json
try:
    summary = json.load(open('/media/andy/Data/tng/mangia_flat/wave1_run_summary.json'))
    n_ok = sum(1 for r in summary if r['status'] == 'ok')
    print(f'Oleada 1 procesada (dataset_entry.h5): {n_ok}/{len(summary)}')
    for r in summary:
        if r['status'] != 'ok':
            print('  NO OK:', r)
except Exception as e:
    print('No pude leer wave1_run_summary.json:', e)
PYEOF

echo "=== $(date) ORQUESTADOR TERMINADO. Ver /media/andy/Data/tng/orchestrator_logs/FINAL_STATUS.txt ==="
cat /media/andy/Data/tng/orchestrator_logs/FINAL_STATUS.txt
