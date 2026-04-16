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

## Quick start

1. Edita `.env` si quieres usar otro directorio de entrada o salida.
2. Crea el entorno e instala dependencias:

```bash
bash scripts/bootstrap.sh
```

3. Lanza reconstruccion batch:

```bash
bash scripts/run_batch.sh
```

4. Lanza el visor web:

```bash
bash scripts/run_viewer.sh
```

## Variables clave de .env

- `RSS_INPUT_DIR`: directorio con los RSS.
- `OUTPUT_DIR`: directorio de cubos reconstruidos.
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

## Procedencia del codigo oficial

La fuente upstream vendorizada corresponde al commit registrado en:

- `official_mangia/UPSTREAM_COMMIT.txt`
