# ImagesMangGenerator

Proveedor de imagenes `g/r/i` para la rama fotometrica del pipeline.

El modulo principal vive en `phase_input/image_provider.py` y produce archivos
NPZ con:

- `image`: tensor `(3, H, W)` en orden `g, r, i`;
- `band_names`;
- `metadata`: JSON con procedencia, unidad, PSF y estado de alineamiento WCS.

## Bootstrap de datos piloto

Los FITS piloto no se versionan. Para copiarlos desde el workspace historico:

```bash
python -m ImagesMangGenerator.phase_input.build_images bootstrap-pilot
```

Esto copia, si existen:

- `TNG50-87-141934-0-127.cube.fits.gz`;
- `TNG50-87-141934-0-127.manga_logcube_74x74.fits.gz`;
- `manga-7443-12703-LOGCUBE.fits.gz`.

Los destinos quedan bajo `ImagesMangGenerator/data/pilot/`, ignorados por Git.

## MaNGIA sintetico

Para generar una imagen piloto desde un cubo MaNGIA:

```bash
python -m ImagesMangGenerator.phase_input.build_images one-mangia \
  --cube ImagesMangGenerator/data/pilot/mangia/TNG50-87-141934-0-127.cube.fits.gz \
  --galaxy-id TNG50-87-141934-0-127 \
  --view-id 0 \
  --outdir ImagesMangGenerator/data/output/mangia
```

Para procesar un directorio de cubos:

```bash
python -m ImagesMangGenerator.phase_input.build_images catalog-mangia \
  --input-dir ImagesMangGenerator/data/pilot/mangia \
  --output-dir ImagesMangGenerator/data/output/mangia \
  --pattern "*.cube.fits.gz" \
  --output-shape native \
  --workers 1
```

Para una ejecucion portable en otra maquina:

```bash
git pull
bash ImagesMangGenerator/scripts/bootstrap_env.sh
cp ImagesMangGenerator/.env.example ImagesMangGenerator/.env
# Editar MANGIA_INPUT_DIR y MANGIA_OUTPUT_DIR
bash ImagesMangGenerator/scripts/run_mangia_catalog.sh
```

El batch MaNGIA acepta:

```bash
--input-dir /ruta/a/cubos
--output-dir /ruta/a/salida
--pattern "*.cube.fits.gz"
--recursive
--skip-existing
--output-shape native   # preserva 69x69 o 74x74 segun el cubo
--output-shape 74,74    # fuerza una grilla comun
```

## MaNGA real desde SDSS

El modo MaNGA usa `drpall_row` para obtener `plateifu`, `objra`, `objdec` e
`ifudesignsize`, descarga cutouts SDSS DR17, los cachea y los reproyecta al WCS
del cubo IFU. Tambien cachea las bandas ya alineadas al WCS del cubo, usando
una clave derivada de WCS + shape.

Ejemplo de una galaxia:

```bash
python -m ImagesMangGenerator.phase_input.build_images one-manga \
  --cube ImagesMangGenerator/data/pilot/manga/manga-7443-12703-LOGCUBE.fits.gz \
  --plateifu 7443-12703 \
  --ra 229.525575871 \
  --dec 42.7458424664 \
  --ifusize 127 \
  --outdir ImagesMangGenerator/data/output/manga \
  --cache-dir ImagesMangGenerator/data/cache/sdss
```

Para catalogo completo se necesita un `drpall-v3_1_1.fits` local:

```bash
python -m ImagesMangGenerator.phase_input.build_images catalog-manga \
  --drpall /ruta/a/drpall-v3_1_1.fits \
  --cubes-dir /ruta/a/logcubes \
  --outdir ImagesMangGenerator/data/output/manga \
  --cache-dir ImagesMangGenerator/data/cache/sdss \
  --workers 4
```

No encontre `drpall` en `/home/andy/pythonprojects/cubes`; debe descargarse o
proveerse antes del catalogo MaNGA completo.

## Credenciales

No se requieren credenciales para SDSS DR17/SkyServer/SAS publico. Si el modo
MaNGA corre con cache frio, si requiere acceso de red.

`TNG_API_KEY` no se usa en este modulo.

## Tests

Smoke test local:

```bash
pytest ImagesMangGenerator/test_smoke.py
```

El test crea un cubo sintetico pequeno, valida la sintesis `g/r/i` y verifica
la persistencia NPZ. Las descargas SDSS no se ejecutan en el smoke test para que
sea reproducible sin red.

## Visualizacion

El `.npz` es el formato cientifico/intermedio: conserva el tensor `float32`, las
unidades y metadatos sin normalizar ni recortar los valores como haria un PNG.
Para mirar una imagen, generar un preview RGB:

```bash
python -m ImagesMangGenerator.phase_input.view_npz_image \
  ImagesMangGenerator/data/output/mangia/TNG50-87-141934-0-127_v0.npz \
  --out ImagesMangGenerator/data/output/previews/TNG50-87-141934-0-127_v0.png
```

El PNG exportado no conserva la grilla cruda de spaxels: por defecto aplica un
suavizado gaussiano leve y reescala el lado mayor a `768 px` para que el preview
sea mas parecido a una imagen astronomica. El NPZ original no se modifica.

Si se omite `--out`, intenta abrir una ventana interactiva con matplotlib.

Opciones utiles:

```bash
--stretch asinh    # default, bueno para galaxias
--stretch sqrt
--stretch linear
--percentile 99.5  # controla el corte alto de brillo
--smooth-sigma 0.85
--preview-size 768
--rgb-order irg    # i->R, r->G, g->B
```

Para generar previews PNG de todos los NPZ de una salida:

```bash
bash ImagesMangGenerator/scripts/render_previews.sh
```

El script lee `ImagesMangGenerator/.env`. Si `PREVIEW_INPUT_DIR` esta vacio,
usa `MANGIA_OUTPUT_DIR`; si `PREVIEW_OUTPUT_DIR` esta vacio, escribe en
`$PREVIEW_INPUT_DIR/previews`.

## Limitaciones v1

- La integracion con Packer queda fuera de alcance porque no existe aun el
  modulo Packer ni `10_dataset.md` en este repo.
- El modo MaNGA asume que las imagenes SDSS calibradas estan en nanomaggies por
  pixel al entrar a la etapa de reproyeccion.
- Si SDSS/WCS falla y `on_wcs_failure=fallback_synthesis`, la salida se genera
  desde el cubo y queda marcada como `source=synthesized` y
  `wcs_aligned=False`.
