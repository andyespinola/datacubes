# Deploy MaNGIA 10k

Proyecto limpio y portable para reconstruir cubos `manga-like` a partir de RSS usando el flujo oficial de MaNGIA, y para visualizar los resultados en un navegador.

## Que incluye

- `rss_to_cube_mangia_official.py`: wrapper local del `regrid()` oficial.
- `batch_reconstruct.py`: runner por lotes para miles de RSS.
- `cube_web_viewer.py`: visor web local para explorar cubos y espectros por spaxel.
- `official_mangia/`: fuente upstream vendorizada y assets minimos requeridos.
- `MaNGIA_catalog.fits`: catalogo para resolver `re_kpc` automaticamente.
- `.env`: configuracion unica para la maquina destino.

## Nota sobre GPU

La maquina destino puede tener GPU, pero el pipeline oficial actual no usa CUDA ni aceleracion por GPU en la reconstruccion. La forma realista de escalar a 10 mil cubos es:

- usar una maquina con buena CPU y bastante RAM;
- ejecutar varios RSS en paralelo;
- fijar `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, etc. para evitar sobreasignacion.

## Estructura esperada

Puedes dejar los RSS dentro de `./rss_input`, o apuntar `RSS_INPUT_DIR` a otro disco en `.env`.

Los cubos reconstruidos se escriben por defecto en `./output`.

## Configuracion en otra maquina

### 1. Que necesitas antes de empezar

- Python 3.10 o superior.
- El repo clonado.
- Acceso al directorio donde estan los 10000 RSS.
- Una fuente real para:
  - `MaNGIA_catalog.fits`
  - `MaStar_CB19.slog_1_5.fits.gz`

### 2. Crear el archivo `.env`

Usa la plantilla incluida:

```bash
cp .env.example .env
```

Ajusta como minimo estas variables:

- `RSS_INPUT_DIR`: carpeta real donde estan los RSS.
- `OUTPUT_DIR`: carpeta donde quieres escribir los cubos.
- `LOG_DIR`: carpeta para logs CSV.
- `CATALOG_SOURCE_PATH` o `CATALOG_URL`: fuente de `MaNGIA_catalog.fits`.
- `TEMPLATE_SOURCE_PATH` o `TEMPLATE_URL`: fuente de `MaStar_CB19.slog_1_5.fits.gz`.
- `BATCH_WORKERS`: cantidad de procesos paralelos.

Ejemplo tipico:

```bash
RSS_INPUT_DIR=/mnt/datos/mangia/rss_10000
OUTPUT_DIR=/mnt/datos/mangia/cubos_reconstruidos
LOG_DIR=/mnt/datos/mangia/logs_reconstruccion

CATALOG_SOURCE_PATH=/mnt/datos/mangia/assets/MaNGIA_catalog.fits
TEMPLATE_SOURCE_PATH=/mnt/datos/mangia/assets/MaStar_CB19.slog_1_5.fits.gz

BATCH_WORKERS=8
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

### 3. Bootstrap del entorno

Este paso:

- crea el virtualenv;
- instala dependencias;
- copia o descarga los assets criticos dentro del deploy;
- crea directorios base de salida y logs.

Comando:

```bash
bash scripts/bootstrap.sh
```

Si todo sale bien, al final deberian existir:

- `./MaNGIA_catalog.fits`
- `./official_mangia/libs/MaStar_CB19.slog_1_5.fits.gz`

### 4. Validacion minima antes de lanzar los 10000

Primero conviene correr una prueba chica:

```bash
bash scripts/run_batch.sh --count 2 --workers 1
```

Revisa que aparezcan salidas como:

- `<prefijo>.cube.fits.gz`
- `<prefijo>.cube_val.fits.gz`

y que en `logs/` o en `LOG_DIR` se haya creado un CSV con estados `ok`.

### 5. Corrida completa

Una vez validado el paso anterior:

```bash
bash scripts/run_batch.sh
```

Si quieres repartir el trabajo entre varios jobs:

```bash
bash scripts/run_batch.sh --start-index 0 --count 2500
bash scripts/run_batch.sh --start-index 2500 --count 2500
bash scripts/run_batch.sh --start-index 5000 --count 2500
bash scripts/run_batch.sh --start-index 7500 --count 2500
```

### 6. Visor local

Para inspeccionar algunos cubos reconstruidos:

```bash
bash scripts/run_viewer.sh
```

Luego abrir `http://127.0.0.1:8000` o el host/puerto configurado en `.env`.

## Quick start

```bash
cp .env.example .env
# editar .env
bash scripts/bootstrap.sh
bash scripts/run_batch.sh --count 2 --workers 1
bash scripts/run_batch.sh
```

## Variables clave de .env

- `RSS_INPUT_DIR`: directorio con los RSS.
- `OUTPUT_DIR`: directorio de cubos reconstruidos.
- `LOG_DIR`: directorio para logs CSV de ejecucion.
- `CATALOG_SOURCE_PATH` / `CATALOG_URL`: de donde obtener `MaNGIA_catalog.fits` al bootstrapear una maquina nueva.
- `TEMPLATE_SOURCE_PATH` / `TEMPLATE_URL`: de donde obtener `MaStar_CB19.slog_1_5.fits.gz`.
- `BATCH_WORKERS`: procesos paralelos.
- `START_INDEX`: offset para repartir trabajo en varios jobs.
- `COUNT`: cuantos RSS procesar desde `START_INDEX`. `0` significa todos.
- `INCLUDE_GAS=1`: suma la componente gaseosa al `PRIMARY`.
- `FORCE_REBUILD=1`: reprocesa aunque existan salidas.

## Ejemplos utiles

Procesar solo los primeros 100 RSS:

```bash
bash scripts/run_batch.sh --count 100
```

Procesar del RSS 2000 al 2999:

```bash
bash scripts/run_batch.sh --start-index 2000 --count 1000
```

Forzar reconstruccion con gas incluido:

```bash
INCLUDE_GAS=1 FORCE_REBUILD=1 bash scripts/run_batch.sh
```

## Logs

Cada corrida batch genera un CSV dentro de `logs/` con:

- estado `ok`, `skipped` o `error`;
- RSS de entrada;
- prefijo de salida;
- tiempo por cubo;
- mensaje de error si corresponde.

## Archivos de salida

Por cada RSS se esperan:

- `<prefijo>.cube.fits.gz`
- `<prefijo>.cube_val.fits.gz`

## Validacion recomendada en otra maquina

Despues del primer deploy, conviene validar con 1 o 2 RSS antes de disparar los 10 mil:

```bash
bash scripts/run_batch.sh --count 2 --workers 1
```

## Fallos comunes

- Si falla el bootstrap por `MaNGIA_catalog.fits`, revisa `CATALOG_SOURCE_PATH` o `CATALOG_URL`.
- Si falla el bootstrap por `MaStar_CB19.slog_1_5.fits.gz`, revisa `TEMPLATE_SOURCE_PATH` o `TEMPLATE_URL`.
- Si el batch no encuentra RSS, revisa `RSS_INPUT_DIR` y `RSS_GLOB`.
- Si la maquina se satura, baja `BATCH_WORKERS` y mantén `OMP_NUM_THREADS=1`.

## Procedencia del codigo oficial

La fuente upstream vendorizada corresponde al commit registrado en:

- `official_mangia/UPSTREAM_COMMIT.txt`
