# TNG50 Galaxia Piloto

Proyecto autocontenido para visualizar la galaxia piloto `TNG50-87-141934-0-127` a partir del `cutout` oficial del subhalo en IllustrisTNG.

## Qué incluye

- viewer web con Flask
- mapas `face-on` y `edge-on`
- vista 3D interactiva WebGL de partículas estelares y gas
- perfiles radiales en el plano estelar
- resumen físico de la galaxia piloto
- bootstrap de datos
- pruebas automáticas con `unittest`

## Datos que usa

- `TNG50-87-141934-0-127.cutout.hdf5`
- `TNG50-87-141934-0-127.subhalo.json`
- `morphs_kinematic_bars.hdf5` opcional, para mostrar barra y fracciones morfológicas globales

Si el repo ya tiene esos archivos en `structural_labeling/cache/`, el bootstrap los copia a esta carpeta. Si no existen, intenta descargarlos desde la API de TNG usando `TNG_API_KEY`.

## Estructura

- `app.py`: servidor principal
- `bootstrap_data.py`: materializa el dataset del piloto
- `pilot_viewer/`: lectura, conversión física y API
- `static/`: frontend del viewer
- `tests/`: suite automatizada

## Puesta en marcha

Comando rápido para ejecutar el proyecto:

```bash
cd /home/andy/pythonprojects/cubes/tng50_galaxia_piloto
./scripts/run_viewer.sh
```

Si necesitás preparar dependencias y materializar los datos antes:

```bash
cd /home/andy/pythonprojects/cubes/tng50_galaxia_piloto
./scripts/bootstrap.sh
./scripts/run_viewer.sh
```

Ejecución paso a paso:

```bash
cd /home/andy/pythonprojects/cubes/tng50_galaxia_piloto
/home/andy/pythonprojects/cubes/.venv/bin/python -m pip install -r requirements.txt
/home/andy/pythonprojects/cubes/.venv/bin/python bootstrap_data.py
/home/andy/pythonprojects/cubes/.venv/bin/python app.py --host 127.0.0.1 --port 5051
```

Luego abrir:

```text
http://127.0.0.1:5051
```

## Tests

```bash
cd /home/andy/pythonprojects/cubes/tng50_galaxia_piloto
/home/andy/pythonprojects/cubes/.venv/bin/python -m unittest discover -s tests -v
```

## Notas físicas del viewer

- Las posiciones del cutout se convierten de `ckpc/h` a `kpc` físicos usando el `redshift` del snapshot.
- Las velocidades se convierten a velocidades peculiares.
- La orientación `face-on` se obtiene a partir del momento angular estelar global.
- La vista `edge-on` usa el mismo sistema rotado, tomando el eje vertical como el del momento angular.
- La vista 3D usa WebGL con buffers binarios y presets de detalle (`rápida`, `equilibrada`, `completa`).
- El modo `completa` intenta cargar todas las partículas dentro de la esfera seleccionada y depende bastante más de la GPU/navegador.
