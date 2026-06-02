# Structural Labeling from TNG to MaNGIA

Este proyecto construye etiquetas estructurales por spaxel para cubos MaNGIA usando la verdad física de TNG50.

## Qué produce

Por galaxia genera:

- `*.labels.npz`
  - `soft_mass[C,H,W]`
  - `soft_light[C,H,W]`
  - `hard_mass[H,W]`
  - `hard_light[H,W]`
  - `confidence_mass[H,W]`
  - `confidence_light[H,W]`
  - `valid_mask[H,W]`
- `*.qa.npz`
  - mapas `face-on`
  - mapas observados
  - residuales para brazos
  - máscara válida
- `*.summary.json`
  - targets globales
  - fracciones recuperadas
  - metadatos de barra

Las clases son:

- `0 no_valido`
- `1 bulbo`
- `2 disco`
- `3 barra`
- `4 brazos`
- `5 other`
- `6 incierto`

## Flujo recomendado

### 1. Construir el manifiesto

```bash
cd /home/andy/pythonprojects/cubes/structural_labeling
/home/andy/pythonprojects/cubes/.venv/bin/python build_manifest.py \
  --catalog /home/andy/pythonprojects/cubes/MaNGIA_catalog.fits \
  --rss-glob "../RSS/*.cube_RSS.fits" \
  --cube-glob "../*.cube.fits.gz" \
  --out manifest.csv
```

### 2. Descargar verdad TNG

```bash
cp .env.example .env
export $(grep -v '^#' .env | xargs)

/home/andy/pythonprojects/cubes/.venv/bin/python download_tng_truth.py \
  --manifest manifest.csv \
  --outdir cache \
  --env-file .env \
  --canonical-id TNG50-87-141934-0-127 \
  --include-gas
```

### 3. Generar etiquetas

```bash
/home/andy/pythonprojects/cubes/.venv/bin/python run_labeling.py \
  --manifest manifest.csv \
  --canonical-id TNG50-87-141934-0-127 \
  --cutout cache/cutouts/TNG50-87-141934-0-127.cutout.hdf5 \
  --metadata cache/metadata/TNG50-87-141934-0-127.subhalo.json \
  --morphology-catalog /ruta/al/catalogo_t_morphology.hdf5 \
  --config default_config.json \
  --outdir outputs
```

Para procesar las unidades ya macheadas por `mangia_asset_matcher`:

```bash
cd /home/aespinola/Documents/pythonprojects/datacubes
python structural_labeling/run_matched_labeling.py \
  --matched-units /home/aespinola/matched_assets/matched_units.csv \
  --catalog MaNGIA_catalog.fits \
  --outdir /media/nuevo/structural_labels \
  --continue-on-error
```

Este runner es reanudable: si existen `*.labels.npz`, `*.qa.npz` y
`*.summary.json`, la unidad se marca como `skipped_existing` salvo que se
use `--overwrite`.

### 3b. Baseline por umbral duro en epsilon

Para responder al baseline metodológico del paper, se puede generar una
segunda carpeta de etiquetas usando solo un umbral duro de circularidad
orbital proxy:

```bash
cd /home/aespinola/Documents/pythonprojects/datacubes
python structural_labeling/run_epsilon_baseline.py \
  --matched-units /home/aespinola/Documents/pythonprojects/datacubes/matched_assets/matched_units.csv \
  --catalog MaNGIA_catalog.fits \
  --outdir /media/nuevo/epsilon_labels \
  --disk-threshold 0.70 \
  --continue-on-error
```

El baseline etiqueta `disco` si `epsilon >= 0.70` y asigna el resto a
`bulbo`, manteniendo la misma grilla, máscara válida, PSF y pesos de masa/luz.
Luego se valida con los mismos comandos de `validation/run_kinematic_validation.py`.

### 4. Calibrar sobre un piloto

```bash
/home/andy/pythonprojects/cubes/.venv/bin/python calibrate_labeling.py \
  --manifest manifest.csv \
  --truth-dir cache \
  --morphology-catalog /ruta/al/catalogo_t_morphology.hdf5 \
  --out-config tuned_config.json \
  --max-samples 20
```

## Qué asume el algoritmo

- Usa `TNG50-1`.
- Usa el catálogo morfológico oficial `(t)` como restricción global, no como máscara por spaxel.
- Usa la misma plantilla SSP local de MaNGIA para construir pesos por luz.
- Usa la vista MaNGIA derivada de `view` y `repeat_count`.
- Si `repeat_count <= 3`, mapea `view=0,1,2` a ejes `x,y,z`.
- Si `repeat_count > 3`, usa un conjunto isotrópico de seis vistas basado en un icosaedro.

## Qué está implementado

- Manifiesto CSV desde `MaNGIA_catalog.fits` y nombres de RSS/cubo.
- Descarga de cutout y metadatos de subhalo vía API de TNG.
- Carga de catálogo morfológico `(t)` desde HDF5 local.
- Pesos por luz por partícula a partir de la librería SSP local.
- Descomposición en:
  - familia `bulbo/disco/other`
  - subdivisión del disco en `disco/barra/brazos`
- Proyección al plano MaNGIA.
- Degradación con PSF y remuestreo a la grilla IFU del cubo.
- `valid_mask` estricto a partir del footprint instrumental y del mapa de señal.
- Separación entre clase física `other` y estado de baja confianza `incierto`.
- Hard labels con suavizado espacial y limpieza de componentes pequeñas.
- Rebalanceo central `bulbo vs disco` antes de endurecer etiquetas.
- Salidas suaves y duras por spaxel.
- Calibración simple por búsqueda en rejilla para piloto.

## Limitaciones actuales

- El downloader del catálogo morfológico `(t)` requiere que ya conozcas su ruta local o la URL directa autenticada.
- Los mapas `pyPipe3D` están contemplados en el manifiesto y en el uso downstream, pero esta primera implementación no los usa todavía dentro del algoritmo de etiquetado.
- La segmentación de brazos se basa en residuales coherentes sobre el disco proyectado; no asume un catálogo oficial de brazos ya listo.

## Smoke test

```bash
/home/andy/pythonprojects/cubes/.venv/bin/python test_smoke.py
```
