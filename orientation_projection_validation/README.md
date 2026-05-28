# Inter-Orientation Projection Validation

Proyecto autocontenido para generar las cuatro proyecciones `q000`, `q045`, `q090`, `q135` requeridas por la validación interna 8.1 del documento metodológico.

El proyecto usa `structural_labeling` como base física, pero no reutiliza su cache. El piloto descarga de nuevo los cutouts TNG en `cache/pilot_fresh/`.

## Flujo Piloto

```bash
cd /home/andy/pythonprojects/datacubes/orientation_projection_validation
cp .env.example .env
# Editar .env con TNG_API_KEY.
# TNG_MORPHOLOGY_CATALOG_URL ya apunta al catálogo oficial:
# https://www.tng-project.org/api/TNG50-1/files/morphs_kinematic_bars.hdf5
# Fuente: https://www.tng-project.org/data/docs/specifications/#sec5t

/home/andy/pythonprojects/datacubes/.venv/bin/python bootstrap_pilot.py \
  --max-galaxies 10 \
  --include-gas \
  --force-download \
  --env-file .env \
  --out-cache cache/pilot_fresh \
  --config default_config.json
```

Luego ejecutar la validación:

```bash
/home/andy/pythonprojects/datacubes/.venv/bin/python run_projection_validation.py \
  --manifest data/pilot_manifest.csv \
  --cache cache/pilot_fresh \
  --config default_config.json \
  --outdir outputs \
  --continue-on-error
```

Y resumir resultados:

```bash
/home/andy/pythonprojects/datacubes/.venv/bin/python summarize_metrics.py \
  --metrics-glob "outputs/*/metrics.json" \
  --out outputs/catalog_interorientation_summary.csv
```

## Descarga Completa

Para preparar el manifiesto completo:

```bash
cd /home/andy/pythonprojects/datacubes/orientation_projection_validation

/home/andy/pythonprojects/datacubes/.venv/bin/python build_projection_manifest.py \
  --catalog /home/andy/pythonprojects/datacubes/MaNGIA_catalog.fits \
  --out data/projection_manifest.csv \
  --config default_config.json
```

Para preparar solo las galaxias únicas que ya pasaron por
`mangia_asset_matcher`:

```bash
cd /home/aespinola/Documents/pythonprojects/datacubes/orientation_projection_validation

python build_projection_manifest.py \
  --matched-units /home/andy/matched_assets/matched_units.csv \
  --out data/projection_manifest_matched.csv \
  --config default_config.json
```

Para iniciar la descarga de todas las galaxias en la máquina de descarga, guardando la cache pesada fuera del repo:

```bash
PYTHONUNBUFFERED=1 /home/andy/pythonprojects/datacubes/.venv/bin/python download_tng_assets.py \
  --manifest data/projection_manifest.csv \
  --out-cache /media/nuevo/tng_cutouts \
  --env-file .env \
  --include-gas \
  --continue-on-error \
  --retry-failures 5 \
  --retry-delay-seconds 300
```

Este comando:

- salta archivos ya descargados y validados;
- escribe `/media/nuevo/tng_cutouts/download_state.jsonl`;
- deja `*.part` solo mientras una descarga está en curso;
- reintenta solo las galaxias fallidas al final de cada pasada.

Para dejarlo corriendo en segundo plano:

```bash
nohup bash -lc 'cd /home/andy/pythonprojects/datacubes/orientation_projection_validation && PYTHONUNBUFFERED=1 /home/andy/pythonprojects/datacubes/.venv/bin/python download_tng_assets.py --manifest data/projection_manifest.csv --out-cache /media/nuevo/tng_cutouts --env-file .env --include-gas --continue-on-error --retry-failures 5 --retry-delay-seconds 300' \
  > download_full.log 2>&1 &
```

Monitoreo:

```bash
tail -f /home/andy/pythonprojects/datacubes/orientation_projection_validation/download_full.log
tail -f /media/nuevo/tng_cutouts/download_state.jsonl
du -sh /media/nuevo/tng_cutouts
```

## Salidas

Por galaxia:

- `outputs/{galaxy_id}/projections.h5`
- `outputs/{galaxy_id}/metrics.json`
- `outputs/{galaxy_id}/qa_mosaic.png`

Resumen global:

- `outputs/catalog_interorientation_summary.csv`

## Métrica 8.1

Para cada clase física se calcula la IoU probabilística entre pares de orientaciones, después de compensar la rotación de cada mapa a una referencia común:

```text
IoU_c(qa, qb) = sum_s min(Y_qa(s,c), Y_qb(s,c)) / sum_s max(Y_qa(s,c), Y_qb(s,c))
```

Con cuatro orientaciones hay seis pares. El promedio por clase da `C_c`, y el promedio de clases da `Cglobal`.

## Notas

- `MaNGIA_catalog.fits` se lee por defecto desde `/home/andy/pythonprojects/datacubes/MaNGIA_catalog.fits`.
- El template SSP se lee por defecto desde `/home/andy/pythonprojects/datacubes/kinematic_moments/templates/MaStar_CB19.slog_1_5.fits.gz`.
- La unidad de descarga es la galaxia TNG única `(snapshot, subhalo_id)`, no la fila/proyección MaNGIA.
- La grilla de validación es MaNGIA-like: `69x69`, `0.5 arcsec/spaxel`, `PSF=1.43 arcsec`.
- No se generan cubos espectrales completos.
