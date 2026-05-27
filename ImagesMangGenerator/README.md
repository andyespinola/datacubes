# ImagesMangGenerator

Guia paso a paso para generar imagenes `g/r/i` a partir de cubos
espectrales MaNGIA o MaNGA.


## 1. Idea general

Un cubo espectral IFU es un arreglo 3D:

```text
flux[lambda, y, x]
```

- `lambda`: eje espectral, es decir, longitud de onda.
- `y, x`: posicion espacial dentro de la galaxia.
- `flux`: intensidad medida o simulada para esa longitud de onda y posicion.

Una imagen RGB normal es un arreglo 3D distinto:

```text
image[channel, y, x]
```

donde `channel` suele tener 3 planos. En este proyecto esos planos son bandas
fotometricas astronomicas:

```text
channel 0 = g
channel 1 = r
channel 2 = i
```

Entonces, el problema que resuelve este modulo es:

```text
cubo espectral FITS  ->  tensor de imagen g/r/i  ->  preview PNG opcional
```

El archivo cientifico principal que produce el pipeline es `.npz`, no `.png`.
El `.npz` conserva los valores numericos de las tres bandas. El PNG se usa solo
para inspeccion visual rapida.

## 2. Donde esta cada cosa

```text
ImagesMangGenerator/
  phase_input/
    image_provider.py       # Logica principal de conversion cubo -> imagen
    build_images.py         # CLI para correr una imagen o un catalogo
    view_npz_image.py       # Convierte NPZ a PNG de preview
  scripts/
    run_mangia_catalog.sh   # Ejecuta catalogos usando .env
    render_previews.sh      # Genera PNG para muchos NPZ
    run_preview_viewer.sh   # Abre el visualizador web de previews
  data/
    pilot/                  # Cubos piloto locales, no versionados
    output/                 # Salidas locales, no versionadas
```

## 3. Entradas y salidas

### Entrada principal

Un cubo FITS, por ejemplo:

```text
TNG50-87-141934-0-127.cube.fits.gz
```

El codigo espera encontrar un arreglo 3D de flujo. Puede estar en:

- la extension primaria del FITS; o
- una extension llamada `FLUX`.

Tambien necesita conocer el eje de longitud de onda. Lo lee desde:

- una extension `WAVE`, si existe; o
- keywords WCS del header como `CRVAL3`, `CRPIX3`, `CDELT3`.

### Salida cientifica

Un archivo `.npz` comprimido con:

```text
image       # arreglo float32 con shape (3, H, W)
band_names  # ["g", "r", "i"]
metadata    # JSON con procedencia, unidad y flags de alineamiento
```

Ejemplo de nombre:

```text
TNG50-87-141934-0-127_v0.npz
```

### Salida visual opcional

Un PNG de preview:

```text
TNG50-87-141934-0-127_v0.png
```

Este PNG no reemplaza al `.npz`: aplica normalizacion, stretch visual,
suavizado y reescalado para que la imagen sea agradable de mirar.

## 4. Que hace el algoritmo

### Caso A: MaNGIA sintetico desde el cubo

Este es el flujo mas directo.

1. Lee el cubo FITS.
2. Obtiene `flux[lambda, y, x]` y el vector `wave[lambda]`.
3. Carga filtros fotometricos `sdss2010-g`, `sdss2010-r`, `sdss2010-i`.
4. Para cada filtro, interpola su curva de respuesta al eje `wave` del cubo.
5. Para cada pixel espacial `(y, x)`, integra el espectro ponderado por el
   filtro.
6. Convierte el resultado a `nanomaggie` por defecto.
7. Apila las tres bandas en un tensor `(3, H, W)`.
8. Opcionalmente recorta o rellena al centro para obtener una forma comun.
9. Guarda el resultado como `.npz`.

En pseudocodigo:

```python
flux, wave = read_cube("galaxy.cube.fits.gz")
bands = []

for filter_name in ["g", "r", "i"]:
    response = interpolate_filter_response(filter_name, wave)
    band_image = integrate(flux * response, axis="lambda")
    bands.append(band_image)

image = stack(bands)  # shape (3, H, W)
save_npz(image)
```

### Caso B: MaNGA real desde SDSS

Este modo no sintetiza la imagen desde el espectro si todo sale bien. En cambio:

1. Usa datos de catalogo (`drpall`) para conocer `plateifu`, RA, DEC y tamano
   del IFU.
2. Descarga imagenes SDSS reales en bandas `g`, `r`, `i`.
3. Recorta la region alrededor de la galaxia.
4. Reproyecta cada banda al WCS y shape espacial del cubo MaNGA.
5. Cachea las bandas descargadas y las bandas ya alineadas.
6. Guarda el tensor `(3, H, W)` como `.npz`.

Si falla la descarga o la reproyeccion, el comportamiento por defecto es usar
`fallback_synthesis`: sintetiza la imagen desde el cubo, igual que en el modo
MaNGIA, y marca `wcs_aligned=False` en los metadatos.

## 5. Preparar el entorno

Desde la raiz del repo:

```bash
cd /home/andy/pythonprojects/datacubes
```

Instalar dependencias si aun no estan instaladas:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Comprobar que el modulo importa y que el smoke test pasa:

```bash
.venv/bin/python -m pytest ImagesMangGenerator
```

## 6. Probar con datos piloto

Los FITS piloto no se versionan. Si existen en el workspace historico, se
pueden copiar con:

```bash
.venv/bin/python -m ImagesMangGenerator.phase_input.build_images bootstrap-pilot
```

Esto intenta copiar:

```text
ImagesMangGenerator/data/pilot/mangia/TNG50-87-141934-0-127.cube.fits.gz
ImagesMangGenerator/data/pilot/mangia/TNG50-87-141934-0-127.manga_logcube_74x74.fits.gz
ImagesMangGenerator/data/pilot/manga/manga-7443-12703-LOGCUBE.fits.gz
```

Si no aparecen, no es un error del pipeline: simplemente esos FITS no estan en
la maquina actual y hay que proveerlos manualmente.

## 7. Generar una imagen MaNGIA desde un cubo

Ejemplo con un solo cubo:

```bash
.venv/bin/python -m ImagesMangGenerator.phase_input.build_images one-mangia \
  --cube ImagesMangGenerator/data/pilot/mangia/TNG50-87-141934-0-127.cube.fits.gz \
  --galaxy-id TNG50-87-141934-0-127 \
  --view-id 0 \
  --outdir ImagesMangGenerator/data/output/mangia \
  --output-shape native
```

Salida esperada:

```text
ImagesMangGenerator/data/output/mangia/TNG50-87-141934-0-127_v0.npz
```

`--output-shape native` conserva la grilla espacial original del cubo. Tambien
puedes forzar una forma comun:

```bash
--output-shape 69,69
--output-shape 74,74
```

Si el cubo es mas grande, se recorta al centro. Si es mas chico, se rellena con
ceros alrededor.

## 8. Generar un catalogo MaNGIA completo

Para correr muchos cubos, primero copia el archivo de configuracion:

```bash
cp ImagesMangGenerator/.env.example ImagesMangGenerator/.env
```

Edita `ImagesMangGenerator/.env`:

```bash
MANGIA_INPUT_DIR=/ruta/a/cubos_mangia
MANGIA_OUTPUT_DIR=/ruta/a/salida_images_mangia
MANGIA_PATTERN=*.cube.fits.gz
MANGIA_RECURSIVE=0
MANGIA_WORKERS=8
MANGIA_OUTPUT_SHAPE=native
MANGIA_SKIP_EXISTING=1
```

Luego ejecuta:

```bash
bash ImagesMangGenerator/scripts/run_mangia_catalog.sh
```

Ese script arma un comando equivalente a:

```bash
.venv/bin/python -m ImagesMangGenerator.phase_input.build_images catalog-mangia \
  --input-dir "$MANGIA_INPUT_DIR" \
  --output-dir "$MANGIA_OUTPUT_DIR" \
  --pattern "$MANGIA_PATTERN" \
  --workers "$MANGIA_WORKERS" \
  --output-shape "$MANGIA_OUTPUT_SHAPE" \
  --skip-existing
```

Al terminar, deberias ver:

```text
salida_images_mangia/
  manifest.csv
  galaxia_1.npz
  galaxia_2.npz
  ...
```

El `manifest.csv` resume cada intento:

```text
galaxy_id,status,source,wcs_aligned,output_path,message
```

Estados importantes:

- `ok`: la imagen se genero correctamente.
- `skipped`: ya existia y `MANGIA_SKIP_EXISTING=1`.
- `failed`: fallo; revisar la columna `message`.

## 9. Generar imagen MaNGA real desde SDSS

Este modo necesita:

- el cubo MaNGA local;
- RA y DEC de la galaxia;
- `plateifu`;
- acceso de red si la cache SDSS esta fria.

Ejemplo de una galaxia:

```bash
.venv/bin/python -m ImagesMangGenerator.phase_input.build_images one-manga \
  --cube ImagesMangGenerator/data/pilot/manga/manga-7443-12703-LOGCUBE.fits.gz \
  --plateifu 7443-12703 \
  --ra 229.525575871 \
  --dec 42.7458424664 \
  --ifusize 127 \
  --outdir ImagesMangGenerator/data/output/manga \
  --cache-dir ImagesMangGenerator/data/cache/sdss
```

Para un catalogo MaNGA completo se necesita un `drpall-v3_1_1.fits` local:

```bash
.venv/bin/python -m ImagesMangGenerator.phase_input.build_images catalog-manga \
  --drpall /ruta/a/drpall-v3_1_1.fits \
  --cubes-dir /ruta/a/logcubes \
  --outdir ImagesMangGenerator/data/output/manga \
  --cache-dir ImagesMangGenerator/data/cache/sdss \
  --workers 4
```

Notas:

- No se requieren credenciales para SDSS DR17/SkyServer/SAS publico.
- Si no hay red y la cache esta vacia, el modo MaNGA real puede fallar o caer
  al fallback sintetico, segun `--on-wcs-failure`.
- `TNG_API_KEY` no se usa en este modulo.

## 10. Convertir NPZ a PNG de preview

Para mirar una imagen concreta:

```bash
.venv/bin/python -m ImagesMangGenerator.phase_input.view_npz_image \
  ImagesMangGenerator/data/output/mangia/TNG50-87-141934-0-127_v0.npz \
  --out ImagesMangGenerator/data/output/previews/TNG50-87-141934-0-127_v0.png
```

Opciones utiles:

```bash
--stretch asinh    # default, bueno para galaxias
--stretch sqrt
--stretch linear
--percentile 99.5  # corte alto de brillo
--smooth-sigma 0.85
--preview-size 768
--rgb-order irg    # i -> R, r -> G, g -> B
```

Para generar PNG para todos los `.npz` de una carpeta:

```bash
bash ImagesMangGenerator/scripts/render_previews.sh
```

Este script tambien lee `ImagesMangGenerator/.env`.

Variables relevantes:

```bash
PREVIEW_INPUT_DIR=        # si esta vacio, usa MANGIA_OUTPUT_DIR
PREVIEW_OUTPUT_DIR=       # si esta vacio, usa PREVIEW_INPUT_DIR/previews
PREVIEW_PATTERN=*.npz
PREVIEW_RECURSIVE=0
PREVIEW_OVERWRITE=0
PREVIEW_STRETCH=asinh
PREVIEW_PERCENTILE=99.5
PREVIEW_SMOOTH_SIGMA=0.85
PREVIEW_SIZE=768
PREVIEW_RGB_ORDER=irg
```

## 11. Ver los PNG en el navegador

Una vez generados los PNG:

```bash
bash ImagesMangGenerator/scripts/run_preview_viewer.sh \
  --preview-dir ImagesMangGenerator/data/output/previews \
  --port 5052
```

Abrir:

```text
http://127.0.0.1:5052
```

El visor esta preparado para catalogos grandes: el backend indexa los PNG y la
UI los muestra por paginas, en lugar de cargar mas de 10000 imagenes al mismo
tiempo.

## 12. Como inspeccionar un NPZ desde Python

```python
import json
import numpy as np

path = "ImagesMangGenerator/data/output/mangia/TNG50-87-141934-0-127_v0.npz"

with np.load(path) as data:
    image = data["image"]
    band_names = data["band_names"].tolist()
    metadata = json.loads(data["metadata"].item())

print(image.shape)      # (3, H, W)
print(band_names)       # ["g", "r", "i"]
print(metadata)
```

Para acceder a una banda:

```python
g = image[0]
r = image[1]
i = image[2]
```

## 13. Recomendaciones para corridas grandes

- Usar `MANGIA_SKIP_EXISTING=1` para poder reanudar si una corrida se corta.
- Empezar con `MANGIA_LIMIT=10` para validar rutas y permisos.
- Usar `MANGIA_OUTPUT_SHAPE=native` si no necesitas una grilla comun.
- Usar `--workers` con cuidado: mas workers acelera, pero tambien aumenta uso
  de CPU, memoria y disco.
- Guardar el `manifest.csv`; es el registro reproducible de que paso con cada
  cubo.
- Generar PNG despues de los NPZ. Los NPZ son la salida importante para modelos
  o analisis.

## 14. Problemas comunes

### `ModuleNotFoundError: astropy` o `speclite`

Faltan dependencias. Instala:

```bash
.venv/bin/pip install -r requirements.txt
```

### `Could not infer wavelength axis`

El FITS no trae extension `WAVE` ni header suficiente para reconstruir el eje
espectral. Revisar que el cubo tenga `CRVAL3`, `CRPIX3` y `CDELT3` o una
extension `WAVE`.

### `Filter ... does not overlap cube wavelength range`

El rango espectral del cubo no cubre una de las bandas `g/r/i`. El pipeline no
puede integrar esa banda de manera confiable.

### No aparecen PNG en el visor

Verificar:

```bash
find ImagesMangGenerator/data/output/previews -maxdepth 1 -type f -name '*.png' | wc -l
```

Si da `0`, primero correr:

```bash
bash ImagesMangGenerator/scripts/render_previews.sh
```

### El catalogo dice `failed`

Abrir `manifest.csv` y mirar la columna `message`. Suele indicar si falta el
cubo, si el FITS no tiene forma 3D, si no hay eje espectral o si fallo SDSS.

## 15. Resumen del flujo completo

```bash
# 1. Preparar entorno
.venv/bin/pip install -r requirements.txt

# 2. Configurar rutas
cp ImagesMangGenerator/.env.example ImagesMangGenerator/.env
# editar ImagesMangGenerator/.env

# 3. Generar NPZ desde cubos
bash ImagesMangGenerator/scripts/run_mangia_catalog.sh

# 4. Generar PNG de preview
bash ImagesMangGenerator/scripts/render_previews.sh

# 5. Abrir visualizador web
bash ImagesMangGenerator/scripts/run_preview_viewer.sh \
  --preview-dir ImagesMangGenerator/data/output/previews \
  --port 5052
```

Resultado final:

```text
cubos FITS
  -> archivos NPZ con image=(3,H,W)
  -> previews PNG
  -> visualizacion web paginada
```
