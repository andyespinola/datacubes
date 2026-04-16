# Guía de Proyectos del Workspace

## 1. Objetivo de este documento

Este archivo resume los proyectos y scripts principales que existen en este workspace de trabajo.

Para cada uno se indica:

- para qué sirve;
- cómo se ejecuta;
- qué entradas espera;
- qué produce como salida;
- y cuáles son sus funcionalidades principales.

La idea es que este documento sirva como mapa rápido del repositorio.

## 2. Entorno común

En la raíz del workspace existe un entorno virtual local:

- `.venv/`

Y un archivo de dependencias general:

- `requirements.txt`

Dependencias principales actualmente usadas:

- `numpy`
- `astropy`
- `scipy`
- `flask`
- `requests`
- `h5py`

Activación recomendada:

```bash
source /home/andy/pythonprojects/cubes/.venv/bin/activate
```

## 3. Datos y archivos base importantes

Archivos de referencia usados por varios proyectos:

- `MaNGIA_catalog.fits`
  - catálogo local de MaNGIA;
  - se usa para resolver metadatos como `snapshot`, `subhalo_id`, `view` y `re_kpc`.
- `official_mangia/libs/MaStar_CB19.slog_1_5.fits.gz`
  - plantilla SSP usada por el flujo oficial y por el proyecto de etiquetado estructural.
- `TNG50-87-141934-0-127.cube.fits.gz`
  - cubo `MaNGA-like` de ejemplo ya reconstruido.
- `TNG50-87-141934-0-127.cube_val.fits.gz`
  - archivo compañero con mapas intrínsecos/asignados del mock.

Documentos metodológicos ya presentes:

- `RSS_A_CUBO_MANGA_LIKE.md`
- `NOTA_DIRECTOR_MANGIA_MANGA_SEGMENTACION.md`
- `NOTA_DIRECTOR_ETIQUETAS_DESDE_SIMULACION.md`

## 4. Proyecto: reconstrucción simple desde RSS

Archivo principal:

- `rss_to_cube_base.py`

### 4.1. Para qué sirve

Este script reconstruye un cubo 3D a partir de un RSS usando una estrategia simple y práctica.

No pretende reproducir exactamente el producto oficial de MaNGIA. Sirve para:

- inspeccionar el contenido del RSS;
- inferir flujo, eje espectral y posiciones IFU;
- reconstruir un cubo aproximado;
- validar geometría;
- y hacer pruebas rápidas.

### 4.2. Entradas

- un RSS FITS de MaNGIA o compatible;
- opcionalmente:
  - la HDU de flujo;
  - el tamaño de grilla;
  - parámetros de reconstrucción;
  - modo solo inspección.

### 4.3. Salidas

- un cubo FITS 3D aproximado;
- o una inspección de las extensiones y metadatos del RSS.

### 4.4. Cómo correrlo

Inspección de un RSS:

```bash
python /home/andy/pythonprojects/cubes/rss_to_cube_base.py \
  /home/andy/pythonprojects/cubes/RSS/TNG50-87-141934-0-127.cube_RSS.fits \
  --no-reconstruct
```

Reconstrucción simple:

```bash
python /home/andy/pythonprojects/cubes/rss_to_cube_base.py \
  /home/andy/pythonprojects/cubes/RSS/TNG50-87-141934-0-127.cube_RSS.fits \
  --out /home/andy/pythonprojects/cubes/cube_simple_test.fits
```

### 4.5. Funcionalidades principales

- detección de HDUs `FLUX`, `WAVE`, `XPOS`, `YPOS`;
- reconstrucción del eje `wave` desde el header si falta una HDU explícita;
- normalización de la orientación del flujo a `(n_fibers, n_wave)`;
- reconstrucción espacial sencilla sobre una grilla regular.

## 5. Proyecto: reconstrucción oficial MaNGIA

Archivos principales:

- `rss_to_cube_mangia_official.py`
- `official_mangia/sin_ifu_clean.py`

### 5.1. Para qué sirve

Este flujo ejecuta la reconstrucción oficial de MaNGIA a partir de un RSS y produce un cubo `MaNGA-like`.

El wrapper local:

- prepara parámetros;
- resuelve `n_fib`;
- busca `re_kpc` en `MaNGIA_catalog.fits` si no se pasa manualmente;
- y llama al `regrid()` oficial vendorizado.

### 5.2. Entradas

- un archivo RSS FITS;
- el catálogo local `MaNGIA_catalog.fits`;
- la plantilla SSP oficial;
- opcionalmente:
  - `--r-eff`
  - `--n-fib`
  - `--include-gas`
  - `--out-prefix`
  - `--noise-sn`
  - `--noise-radius`

### 5.3. Salidas

Por cada RSS:

- `<prefijo>.cube.fits.gz`
- `<prefijo>.cube_val.fits.gz`

### 5.4. Cómo correrlo

Ejemplo real:

```bash
python /home/andy/pythonprojects/cubes/rss_to_cube_mangia_official.py \
  /home/andy/pythonprojects/cubes/RSS/TNG50-87-141934-0-127.cube_RSS.fits
```

Con gas incluido:

```bash
python /home/andy/pythonprojects/cubes/rss_to_cube_mangia_official.py \
  /home/andy/pythonprojects/cubes/RSS/TNG50-87-141934-0-127.cube_RSS.fits \
  --include-gas \
  --out-prefix /home/andy/pythonprojects/cubes/TNG50-87-141934-0-127-with-gas
```

### 5.5. Funcionalidades principales

- parsing del nombre RSS tipo `TNG50-87-141934-0-127.cube_RSS.fits`;
- inferencia de `n_fib` desde `IFUCON`;
- búsqueda automática de `re_kpc` en `MaNGIA_catalog.fits`;
- wrapper local del código oficial vendorizado;
- salida en formato compatible con el estilo MaNGA.

## 6. Proyecto: visor web local de cubos MaNGIA

Archivos principales:

- `cube_web_viewer.py`
- `cube_viewer_static/index.html`
- `cube_viewer_static/app.js`
- `cube_viewer_static/styles.css`

### 6.1. Para qué sirve

Es un visor web local para explorar cubos reconstruidos.

Permite:

- navegar slices espectrales;
- seleccionar spaxels;
- inspeccionar el espectro completo del spaxel;
- ver `ERROR`, `MASK` y `GAS`;
- y, actualmente, superponer etiquetas estructurales generadas por el proyecto de etiquetado.

### 6.2. Entradas

- cubos FITS 3D dentro del workspace;
- opcionalmente archivos asociados:
  - `*.labels.npz`
  - `*.summary.json`

### 6.3. Salidas

- servidor web local;
- visualización interactiva en navegador;
- inspección de overlays de estructura y probabilidades por spaxel.

### 6.4. Cómo correrlo

```bash
python /home/andy/pythonprojects/cubes/cube_web_viewer.py
```

Abrir en navegador:

```text
http://127.0.0.1:8000
```

### 6.5. Funcionalidades principales

- descubre cubos 3D válidos en el workspace;
- muestra mapa medio o slice espectral;
- muestra espectro por spaxel;
- integra overlays:
  - `soft_mass`
  - `soft_light`
  - `hard_mass` y `hard_light` default
  - `hard_mass_050`, `hard_mass_055`, `hard_mass_060`
  - `hard_light_050`, `hard_light_055`, `hard_light_060`
- muestra resumen global de etiquetas;
- muestra conteos por umbral duro;
- muestra clase dura, variantes por umbral, confianza y probabilidades del spaxel.

## 7. Proyecto: armonización MaNGIA hacia `LOGCUBE-like 74x74`

Directorio:

- `mangia_logcube_74x74/`

Archivos principales:

- `mangia_logcube_74x74/build_mangia_logcube.py`
- `mangia_logcube_74x74/harmonize_logcube.py`
- `mangia_logcube_74x74/validate_logcube.py`
- `mangia_logcube_74x74/default_config.json`
- `mangia_logcube_74x74/README.md`

### 7.1. Para qué sirve

Este proyecto genera una variante adicional del cubo mock para dejarlo comparable con un `LOGCUBE` real de MaNGA.

No reemplaza la reconstrucción oficial de MaNGIA. El flujo es:

1. correr el cubo oficial normal;
2. usar ese cubo como fuente de procedencia;
3. armonizarlo a una salida `LOGCUBE-like` con:
   - geometría `74x74`
   - WCS espacial estilo MaNGA
   - grid espectral del template real
   - estructura FITS `PRIMARY + FLUX + IVAR + MASK + WAVE`

### 7.2. Entradas

- RSS FITS MaNGIA;
- `MaNGIA_catalog.fits`;
- template SSP del flujo oficial;
- `LOGCUBE` real de referencia, normalmente desde `manga_compare_project/data/`.

### 7.3. Salidas

- cubo oficial MaNGIA de procedencia, opcionalmente conservado;
- `<prefijo>.manga_logcube_74x74.fits.gz`
- `<prefijo>.manga_logcube_74x74.summary.json`

### 7.4. Cómo correrlo

```bash
python /home/andy/pythonprojects/cubes/mangia_logcube_74x74/build_mangia_logcube.py \
  /ruta/al/rss.fits \
  --reference-logcube /home/andy/pythonprojects/cubes/manga_compare_project/data/manga-7443-12703-LOGCUBE.fits.gz \
  --outdir /home/andy/pythonprojects/cubes/mangia_logcube_74x74/output \
  --keep-official
```

### 7.5. Funcionalidades principales

- ejecuta el regrid oficial sin modificar el upstream;
- armoniza espacialmente a `74x74`;
- remuestrea espectralmente al grid del `LOGCUBE` real;
- convierte a `FLUX/IVAR/MASK/WAVE`;
- valida WCS, shape, unidades y conservación de flujo;
- actúa como puente entre MaNGIA y MaNGA para comparación y ML.

## 8. Proyecto: etiquetado estructural desde TNG hacia MaNGIA

Directorio:

- `structural_labeling/`

Archivos principales:

- `structural_labeling/build_manifest.py`
- `structural_labeling/download_tng_truth.py`
- `structural_labeling/run_labeling.py`
- `structural_labeling/calibrate_labeling.py`
- `structural_labeling/default_config.json`
- `structural_labeling/test_smoke.py`

### 8.1. Para qué sirve

Este proyecto construye etiquetas estructurales por spaxel usando la verdad física de TNG y la geometría del cubo MaNGIA.

Produce targets para segmentación estructural:

- `bulbo`
- `disco`
- `barra`
- `brazos`
- `other`
- `incierto`
- `no_valido`

### 8.2. Entradas

#### A. Desde MaNGIA

- `MaNGIA_catalog.fits`
- cubo `*.cube.fits.gz`
- manifiesto CSV

#### B. Desde TNG

- `cutout` HDF5 del subhalo;
- metadatos JSON del subhalo;
- catálogo morfológico oficial `(t)` en HDF5;
- API key de TNG para descargas.

#### C. Configuración

- `default_config.json` o un JSON ajustado;
- plantilla SSP.

### 8.3. Salidas

Por galaxia:

- `*.labels.npz`
- `*.qa.npz`
- `*.summary.json`

Contenido principal:

- `soft_mass`
- `soft_light`
- `hard_mass`
- `hard_light`
- `confidence_mass`
- `confidence_light`
- `valid_mask`

### 8.4. Cómo correrlo

#### 1. Construir manifiesto

```bash
cd /home/andy/pythonprojects/cubes/structural_labeling
/home/andy/pythonprojects/cubes/.venv/bin/python build_manifest.py \
  --catalog /home/andy/pythonprojects/cubes/MaNGIA_catalog.fits \
  --rss-glob "../RSS/*.cube_RSS.fits" \
  --cube-glob "../*.cube.fits.gz" \
  --out manifest.csv
```

#### 2. Descargar verdad TNG

```bash
/home/andy/pythonprojects/cubes/.venv/bin/python download_tng_truth.py \
  --manifest manifest.csv \
  --outdir cache \
  --env-file .env \
  --canonical-id TNG50-87-141934-0-127 \
  --include-gas
```

#### 3. Generar etiquetas

```bash
/home/andy/pythonprojects/cubes/.venv/bin/python run_labeling.py \
  --manifest manifest.csv \
  --canonical-id TNG50-87-141934-0-127 \
  --cutout cache/cutouts/TNG50-87-141934-0-127.cutout.hdf5 \
  --metadata cache/metadata/TNG50-87-141934-0-127.subhalo.json \
  --morphology-catalog cache/morphs_kinematic_bars.hdf5 \
  --config default_config.json \
  --outdir outputs
```

#### 4. Smoke test

```bash
cd /home/andy/pythonprojects/cubes/structural_labeling
/home/andy/pythonprojects/cubes/.venv/bin/python test_smoke.py
```

### 8.5. Funcionalidades principales

- construcción de manifiesto desde catálogo MaNGIA;
- descarga de `cutout` y metadatos TNG;
- carga de catálogo morfológico oficial;
- conversión a unidades físicas;
- pesos por masa y por luz;
- descomposición en familias y subestructuras;
- proyección a la vista observacional;
- degradación con PSF y remuestreo IFU;
- `valid_mask` estricto;
- separación entre `other` e `incierto`;
- regularización espacial de hard labels;
- corrección central `bulbo vs disco`.

## 9. Proyecto: comparación con MaNGA real

Directorio:

- `manga_compare_project/`

Archivos principales:

- `manga_compare_project/download_manga_drp.py`
- `manga_compare_project/manga_compare_viewer.py`
- `manga_compare_project/README.md`

### 9.1. Para qué sirve

Permite descargar un producto real de MaNGA DRP y compararlo visualmente con cubos mock `MaNGA-like`.

Soporta:

- `LOGCUBE`
- `LINCUBE`
- `LOGRSS`
- `LINRSS`

### 9.2. Entradas

- `plateifu` de MaNGA;
- tipo de producto;
- release y versión DRP;
- opcionalmente cubos mock copiados a `data/`.

### 9.3. Salidas

- FITS descargado desde el SAS de SDSS;
- visor web comparativo local.

### 9.4. Cómo correrlo

```bash
cd /home/andy/pythonprojects/cubes/manga_compare_project
bash scripts/bootstrap.sh
bash scripts/download_sample.sh
bash scripts/run_viewer.sh
```

Abrir en:

```text
http://127.0.0.1:8010
```

Descargar un RSS real en vez de cubo:

```bash
MANGA_PRODUCT=LOGRSS bash scripts/download_sample.sh
```

### 9.5. Funcionalidades principales

- construcción de URL oficial SAS de SDSS;
- descarga de producto DRP real;
- visor para cubos MaNGA reales;
- visor para RSS MaNGA reales;
- comparación con cubos mock en la misma interfaz.

## 10. Proyecto portable para 10 mil cubos

Directorio:

- `deploy_mangia_10k/`

Archivos principales:

- `deploy_mangia_10k/rss_to_cube_mangia_official.py`
- `deploy_mangia_10k/batch_reconstruct.py`
- `deploy_mangia_10k/cube_web_viewer.py`
- `deploy_mangia_10k/scripts/bootstrap.sh`
- `deploy_mangia_10k/scripts/run_batch.sh`
- `deploy_mangia_10k/scripts/run_viewer.sh`

### 10.1. Para qué sirve

Es un proyecto limpio y portable para ejecutar reconstrucción oficial de MaNGIA en otra máquina, pensado para procesamiento batch de miles de RSS.

### 10.2. Entradas

- RSS en `rss_input/` o en el directorio apuntado por `.env`;
- `MaNGIA_catalog.fits`;
- configuración `.env`.

### 10.3. Salidas

- cubos reconstruidos en `output/`;
- logs CSV en `logs/`;
- visor web local.

### 10.4. Cómo correrlo

```bash
cd /home/andy/pythonprojects/cubes/deploy_mangia_10k
bash scripts/bootstrap.sh
bash scripts/run_batch.sh
```

Visor:

```bash
bash scripts/run_viewer.sh
```

### 10.5. Funcionalidades principales

- procesamiento batch paralelo por CPU;
- control de rango con `--start-index` y `--count`;
- opción de incluir gas;
- logging detallado por RSS;
- despliegue portable con `.env`.

## 11. Código upstream vendorizado

Directorio:

- `official_mangia/`

### 11.1. Para qué sirve

Contiene la fuente oficial vendorizada del flujo de MaNGIA que se usa localmente a través del wrapper.

### 11.2. Entradas y salidas

No está pensado como proyecto de usuario final por sí solo en este workspace.

Se usa internamente desde:

- `rss_to_cube_mangia_official.py`
- `deploy_mangia_10k/rss_to_cube_mangia_official.py`

### 11.3. Archivos clave

- `official_mangia/sin_ifu_clean.py`
- `official_mangia/UPSTREAM_COMMIT.txt`
- `official_mangia/libs/MaStar_CB19.slog_1_5.fits.gz`

## 12. Documentos metodológicos

Archivos:

- `RSS_A_CUBO_MANGA_LIKE.md`
- `NOTA_DIRECTOR_MANGIA_MANGA_SEGMENTACION.md`
- `NOTA_DIRECTOR_ETIQUETAS_DESDE_SIMULACION.md`

### 12.1. Para qué sirven

No son proyectos ejecutables, pero sí documentación clave del workspace.

Cubren:

- explicación del RSS y su conversión a cubo;
- comparación metodológica entre MaNGIA y MaNGA;
- y construcción de etiquetas estructurales desde la simulación.

## 13. Orden sugerido de uso

Si el objetivo es reconstruir un cubo y explorarlo:

1. usar `rss_to_cube_mangia_official.py`
2. abrir `cube_web_viewer.py`

Si el objetivo es construir labels estructurales:

1. reconstruir el cubo oficial
2. si se quiere compatibilidad fuerte con MaNGA, pasar por `mangia_logcube_74x74/`
3. preparar manifiesto en `structural_labeling/`
4. descargar verdad TNG
5. correr `run_labeling.py`
6. inspeccionar resultados en `cube_web_viewer.py`

Si el objetivo es comparar con MaNGA real:

1. usar `manga_compare_project/`
2. descargar `LOGCUBE` o `LOGRSS`
3. si se quiere un mock con geometría compatible, usar `mangia_logcube_74x74/`
4. abrir el visor comparativo

Si el objetivo es procesar miles de RSS en otra máquina:

1. usar `deploy_mangia_10k/`
2. configurar `.env`
3. correr `run_batch.sh`

## 14. Observación final

Este workspace no es un único proyecto monolítico, sino un conjunto de piezas complementarias:

- reconstrucción simple;
- reconstrucción oficial;
- visualización;
- comparación con MaNGA real;
- etiquetado estructural;
- y despliegue batch a gran escala.

La recomendación práctica es tomar este documento como punto de entrada y luego entrar al README específico del subproyecto que se quiera usar.

## 15. Preparación para subir a GitHub

Nombre recomendado del repositorio:

- `datacubes`

La raíz del workspace incluye ahora:

- `README.md` como entrada breve al repositorio;
- `.gitignore` para evitar subir datos pesados o productos generados;
- `MaNGIA_catalog.fits` como excepción explícita, porque el catálogo sí debe quedar disponible.

La política recomendada para el repositorio es:

- subir código, documentación, configuraciones, scripts y assets pequeños;
- no subir cubos, RSS, cutouts, salidas `.npz`, archivos `.hdf5` descargados ni paquetes `.tar.gz`;
- no subir `.env` con rutas locales o claves API;
- conservar solo `.env.example` como plantilla;
- mantener `MaNGIA_catalog.fits` en la raíz del repositorio.

Archivos y carpetas que quedan excluidos por defecto:

- `*.fits` y `*.fits.gz`, excepto `MaNGIA_catalog.fits`;
- `*.h5` y `*.hdf5`;
- `*.npz`;
- `*.tar.gz`;
- `.venv/`;
- `manga_compare_project/data/`;
- `mangia_logcube_74x74/output/`;
- `structural_labeling/cache/`;
- `structural_labeling/outputs/`.

Nota importante: el template SSP `MaStar_CB19.slog_1_5.fits.gz` no se sube porque también es un FITS. Para correr la reconstrucción oficial en una máquina nueva hay que colocarlo manualmente en `official_mangia/libs/` o en la ruta equivalente del proyecto deploy.

Para crear el repositorio privado desde GitHub CLI:

```bash
gh repo create datacubes --private --source=. --remote=origin --push
```

Si se crea manualmente desde la web de GitHub, después ejecutar:

```bash
git remote add origin git@github.com:<usuario>/datacubes.git
git push -u origin main
```
