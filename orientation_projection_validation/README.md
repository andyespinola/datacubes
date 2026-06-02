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

Y resumir resultados. Este paso no regenera proyecciones; solo lee los
`metrics.json` ya producidos y crea un CSV más un reporte Markdown concentrado:

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

Para la validación de las 500 unidades macheadas, usar un manifiesto
deduplicado por galaxia TNG única, generado desde `matched_units.csv`.
Este es el manifiesto correcto para cruzar con la corrida cinemática; no
usar `data/projection_manifest.csv` salvo que se quiera procesar el catálogo
MaNGIA completo.

```bash
cd /home/aespinola/Documents/pythonprojects/datacubes

python -m orientation_projection_validation.build_projection_manifest \
  --matched-units /home/aespinola/Documents/pythonprojects/datacubes/matched_assets/matched_units.csv \
  --out orientation_projection_validation/data/projection_manifest_matched.csv \
  --config orientation_projection_validation/default_config.json
```

Con la selección actual de 500 unidades, este manifiesto debe contener 424
galaxias únicas y `source_rows` debe sumar 500.

Para correr la validación de proyecciones sobre esas galaxias:

```bash
cd /home/aespinola/Documents/pythonprojects/datacubes

python -m orientation_projection_validation.run_projection_validation \
  --manifest orientation_projection_validation/data/projection_manifest_matched.csv \
  --cache /media/nuevo/tng_cutouts \
  --morphology-catalog /media/nuevo/tng_cutouts/morphology/morphs_kinematic_bars.hdf5 \
  --config orientation_projection_validation/default_config.json \
  --outdir /media/nuevo/orientation_projection_validation/outputs_matched \
  --continue-on-error
```

Para el baseline simple de umbral duro en `epsilon`, usar el mismo manifiesto y
cambiar solo el modelo de etiquetas:

```bash
python -m orientation_projection_validation.run_projection_validation \
  --manifest orientation_projection_validation/data/projection_manifest_matched.csv \
  --cache /media/nuevo/tng_cutouts \
  --morphology-catalog /media/nuevo/tng_cutouts/morphology/morphs_kinematic_bars.hdf5 \
  --config orientation_projection_validation/default_config.json \
  --outdir /media/nuevo/orientation_projection_validation/outputs_matched_epsilon \
  --label-model epsilon \
  --epsilon-threshold 0.70 \
  --continue-on-error
```

Y resumir resultados. Este paso no regenera proyecciones; solo lee los
`metrics.json` ya producidos y crea un CSV más un reporte Markdown concentrado:

```bash
cd /home/aespinola/Documents/pythonprojects/datacubes/orientation_projection_validation

python summarize_metrics.py \
  --metrics-glob "/media/nuevo/orientation_projection_validation/outputs_matched/*/metrics.json" \
  --out /media/nuevo/orientation_projection_validation/catalog_interorientation_summary_matched.csv \
  --report /media/nuevo/orientation_projection_validation/catalog_interorientation_summary_matched.md
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
